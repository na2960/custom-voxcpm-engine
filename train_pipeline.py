import os
import json
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from voxcpm import VoxCPM
from peft import LoraConfig, get_peft_model, SafetensorsStorage

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
BATCH_SIZE = 2                 # Keep low to prevent CUDA OOM on T4 GPUs
GRADIENT_ACCUMULATION_STEPS = 4 # Simulates a virtual batch size of 8 (2 * 4)
LEARNING_RATE = 2e-4           # Standard stable learning rate for LoRA tuning
LORA_RANK = 32                 # Dimensional rank of the injected matrix tensors
LORA_ALPHA = 32                # Scaling factor for the adapter weights
EPOCHS = 3                     # Total passes over your custom voice dataset
DATA_DIR = "./my-voice-dataset"
OUTPUT_DIR = "./checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CUSTOM PYTORCH DATA LOADER
# ==========================================
class VoiceDataset(Dataset):
    def __init__(self, data_dir, sample_rate=24000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.samples = []
        
        manifest_path = os.path.join(data_dir, "train_manifest.jsonl")
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = os.path.join(self.data_dir, sample["audio_path"])
        
        # Load audio file waveform tensor and match model expected sample rate
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Standardize shape to 1D sequence tensor [samples]
        waveform = waveform.mean(dim=0) 
        
        return {
            "waveform": waveform,
            "text": sample["text"]
        }

# Custom collation function to dynamically pad varying audio/text lengths per batch
def collate_fn(batch):
    waveforms = [item["waveform"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Pad audio tensors with zero values so they form a uniform matrix shape
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return {"waveforms": padded_waveforms, "texts": texts}

# ==========================================
# 3. CORE EXECUTABLE TRAINING LOOP
# ==========================================
def run_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}")

    # Initialize Base Model and cast weights to bfloat16 to optimize VRAM
    print("Loading base VoxCPM2 network tensors...")
    voxcpm_wrapper = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    base_model = voxcpm_wrapper.tts_model
    base_model.to(device=device, dtype=torch.bfloat16)

    # Freeze all core upstream layers
    for param in base_model.parameters():
        param.requires_grad = False

    # Inject low-rank adapters into the linear layers of the Diffusion Transformer block
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "to_q", "to_v"], # LocDiT / TSLM attention projections
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Initialize the Optimizer (AdamW handles weight decay cleanly for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Prepare data streaming pipeline
    dataset = VoiceDataset(data_dir=DATA_DIR, sample_rate=base_model.sample_rate)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model.train()
    print("Beginning fine-tuning iterations...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            # 1. Cast input features to execution space
            waveforms = batch["waveforms"].to(device=device, dtype=torch.bfloat16)
            texts = batch["texts"]
            
            # 2. Execute forward pass through the training graph
            # VoxCPM2 computes internal conditional flow-matching loss natively during training execution
            outputs = model(waveforms=waveforms, texts=texts)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            
            # 3. Backward pass to calculate gradients
            loss.backward()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 4. Perform optimizer step based on gradient accumulation settings
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                # Gradient clipping prevents parameters from exploding during matrix multiplication
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            if step % 5 == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS} | Step: {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        # 5. Checkpointing at the end of every epoch
        checkpoint_path = os.path.join(OUTPUT_DIR, f"voxcpm2_lora_epoch_{epoch+1}")
        model.save_pretrained(checkpoint_path)
        print(f"Saved trainable adapter weights tensor to: {checkpoint_path}")

    print("Training complete! The optimized .safetensors matrix is ready for podcast generation.")

if __name__ == "__main__":
    run_training()
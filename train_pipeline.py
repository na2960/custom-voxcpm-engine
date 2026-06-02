# train_pipeline.py
import os
import json
import argparse
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from voxcpm import VoxCPM
from peft import LoraConfig, get_peft_model

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
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        waveform = waveform.mean(dim=0) 
        return {"waveform": waveform, "text": sample["text"]}

def collate_fn(batch):
    waveforms = [item["waveform"] for item in batch]
    texts = [item["text"] for item in batch]
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return {"waveforms": padded_waveforms, "texts": texts}

def train_loop(data_dir, epochs, batch_size, lr, grad_accum, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Targeting active device pipeline: {device}")

    print("Loading base VoxCPM2 network weights...")
    voxcpm_wrapper = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    base_model = voxcpm_wrapper.tts_model
    base_model.to(device=device, dtype=torch.bfloat16)

    for param in base_model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=32, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "to_q", "to_v"], 
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = VoiceDataset(data_dir=data_dir, sample_rate=base_model.sample_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()
    print("Launching fine-tuning passes...")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            waveforms = batch["waveforms"].to(device=device, dtype=torch.bfloat16)
            texts = batch["texts"]
            
            outputs = model(waveforms=waveforms, texts=texts)
            loss = outputs.loss / grad_accum
            
            loss.backward()
            total_loss += loss.item() * grad_accum
            
            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            if step % 2 == 0:
                print(f"Epoch: {epoch}/{epochs} | Step: {step}/{len(dataloader)} | Flow Loss: {loss.item() * grad_accum:.4f}")
        
        checkpoint_path = os.path.join(output_dir, f"voxcpm2_lora_epoch_{epoch}")
        model.save_pretrained(checkpoint_path)
        print(f"Successfully saved tracking checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom VoxCPM2 engine via repository scripts.")
    parser.add_argument("data_dir", help="Directory holding the train_manifest.jsonl file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", default="./checkpoints", help="Path to save adapters")
    args = parser.parse_args()
    
    train_loop(args.data_dir, args.epochs, args.batch_size, args.lr, args.grad_accum, args.output_dir)

# custom-voxcpm-engine

A minimal custom training engine scaffold for a VoxCPM-style model.

Structure:

custom-voxcpm-engine/
├── configs/
│   └── lora_config.json        # Defines rank (r), alpha, and target modules
├── modules/
│   ├── __init__.py
│   └── custom_loss.py          # Where you manipulate the training tensor loss math
├── dataset/
│   └── prepare_data.py         # Converts raw .wav voice clips to Mel-spectrogram tensors
├── train_pipeline.py           # Core PyTorch training loop executable
└── README.md

## Usage

1. Convert .wav audio to Mel tensor inputs:
   ```bash
   python dataset/prepare_data.py raw_wavs/ prepared_tensors/
   ```

2. Train the model:
   ```bash
   python train_pipeline.py prepared_tensors/ --epochs 10 --batch_size 4
   ```

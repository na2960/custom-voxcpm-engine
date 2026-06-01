import os
import json
import argparse
import torchaudio

def validate_and_build_manifest(raw_data_dir, output_dir, target_sr=24000):
    """
    Scans a raw data directory for paired .wav and .txt files, verifies audio health,
    moves them to a structured target layout, and creates the JSONL manifest tensor map.
    """
    clips_target_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_target_dir, exist_ok=True)
    
    manifest_entries = []
    skipped_count = 0
    processed_count = 0

    print(f"Scanning raw files in: {raw_data_dir}")
    
    # Locate all distinct file stems by scanning for .wav entries
    all_files = os.listdir(raw_data_dir)
    wav_stems = [os.path.splitext(f)[0] for f in all_files if f.endswith(".wav")]

    if not wav_stems:
        print("Error: No .wav files discovered in the target directory.")
        return

    for stem in sorted(wav_stems):
        wav_filename = f"{stem}.wav"
        txt_filename = f"{stem}.txt"
        
        raw_wav_path = os.path.join(raw_data_dir, wav_filename)
        raw_txt_path = os.path.join(raw_data_dir, txt_filename)
        
        # Guard: Check for missing matching transcript file
        if not os.path.exists(raw_txt_path):
            print(f"Warning: Transcript missing for {wav_filename}. Skipping clip.")
            skipped_count += 1
            continue
            
        try:
            # Metadata Check: Ensure PyTorch can read the audio tensor safely
            info = torchaudio.info(raw_wav_path)
            
            # Read and sanitize text transcript
            with open(raw_txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                
            if not transcript:
                print(f"Warning: Transcript for {stem} is empty. Skipping clip.")
                skipped_count += 1
                continue
            
            # Establish the target path names inside your clean dataset architecture
            relative_audio_path = os.path.join("clips", wav_filename)
            final_wav_path = os.path.join(output_dir, relative_audio_path)
            
            # If directories match, we bypass copies; otherwise, transfer the asset
            if os.path.abspath(raw_wav_path) != os.path.abspath(final_wav_path):
                # Using torchaudio to load and save handles any low-level header cleanup automatically
                waveform, sr = torchaudio.load(raw_wav_path)
                torchaudio.save(final_wav_path, waveform, sr)
            
            # Construct the exact dictionary record expected by the training loop loader
            manifest_entry = {
                "audio_path": relative_audio_path,
                "text": transcript
            }
            manifest_entries.append(manifest_entry)
            processed_count += 1
            
        except Exception as e:
            print(f"Error: Waveform file {wav_filename} is corrupted. Reason: {e}. Skipping.")
            skipped_count += 1
            continue

    # Write the completed manifest map to the destination directory
    manifest_output_path = os.path.join(output_dir, "train_manifest.jsonl")
    with open(manifest_output_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("\n" + "="*50)
    print("DATA EXTRACTION PROFILE COMPLETE")
    print("="*50)
    print(f"Successfully Structured: {processed_count} clips")
    print(f"Flagged/Skipped:        {skipped_count} clips")
    print(f"Manifest Generated At:  {manifest_output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw voice datasets for VoxCPM2 training loops.")
    parser.add_argument("raw_data_dir", help="Directory holding your unorganized matching .wav and .txt files")
    parser.add_argument("--output_dir", default="./my-voice-dataset", help="Destination path for training pipeline")
    args = parser.parse_args()
    
    validate_and_build_manifest(args.raw_data_dir, args.output_dir)
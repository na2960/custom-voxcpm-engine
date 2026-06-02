# prepare_data.py
import os
import json
import argparse
import torchaudio

def main():
    parser = argparse.ArgumentParser(description="Match GitHub text files with Kaggle audio files.")
    parser.add_argument("kaggle_audio_dir", help="Path to unpacked Kaggle audio dataset")
    parser.add_argument("--github_text_dir", default="./", help="Path to cloned GitHub text files")
    parser.add_argument("--output_dir", default="./my-voice-dataset", help="Target training folder")
    args = parser.parse_args()

    clips_target_dir = os.path.join(args.output_dir, "clips")
    os.makedirs(clips_target_dir, exist_ok=True)
    
    manifest_entries = []
    processed_count = 0

    print(f"Matching text from {args.github_text_dir} with audio from {args.kaggle_audio_dir}...")

    # Scan the GitHub directory for text transcripts
    for filename in sorted(os.listdir(args.github_text_dir)):
        if filename.endswith(".txt"):
            file_stem = os.path.splitext(filename)[0]
            
            repo_txt_path = os.path.join(args.github_text_dir, filename)
            kaggle_wav_path = os.path.join(args.kaggle_audio_dir, f"{file_stem}.wav")
            
            # Verify the matching wave tensor file exists inside Kaggle's input
            if os.path.exists(kaggle_wav_path):
                try:
                    with open(repo_txt_path, "r", encoding="utf-8") as f:
                        transcript_text = f.read().strip()
                    
                    relative_audio_path = os.path.join("clips", f"{file_stem}.wav")
                    final_wav_destination = os.path.join(args.output_dir, relative_audio_path)
                    
                    # Read waveform arrays and save to the clean output directory
                    waveform_tensor, sample_rate = torchaudio.load(kaggle_wav_path)
                    torchaudio.save(final_wav_destination, waveform_tensor, sample_rate)
                    
                    # Log mapping entry for the DataLoader manifest
                    manifest_record = {
                        "audio_path": relative_audio_path,
                        "text": transcript_text
                    }
                    manifest_entries.append(manifest_record)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Skipping stem {file_stem} due to reading error: {e}")

    manifest_output_file = os.path.join(args.output_dir, "train_manifest.jsonl")
    with open(manifest_output_file, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Success! Generated {processed_count} paired samples inside {manifest_output_file}")

if __name__ == "__main__":
    main()

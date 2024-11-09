import argparse
import os

import pandas as pd
import requests

# Set up argument parser
parser = argparse.ArgumentParser(description="Transcribe audio files using ASR API and save results to a CSV.")
parser.add_argument("csv_file_path", type=str, help="Path to the CSV file.")
parser.add_argument("audio_folder_path", type=str, help="Path to the folder containing audio files.")
parser.add_argument("output_csv_path", type=str, help="Path to save the updated CSV file with transcriptions.")
args = parser.parse_args()

# Load dataset CSV
df = pd.read_csv(args.csv_file_path)

# Define API endpoint
url = "http://localhost:8001/asr"

# Process each audio file
transcriptions = []
for _, row in df.iterrows():
    audio_path = os.path.join(args.audio_folder_path, row["filename"])
    
    # Check if audio file exists
    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}")
        transcriptions.append("")
        continue

    # Send audio file to the ASR API for transcription
    with open(audio_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            transcriptions.append(data["transcription"])
        else:
            print(f"Error transcribing file {audio_path}: {response.text}")
            transcriptions.append("")

# Add transcriptions the dataframe
df["generated_text"] = transcriptions

# Save the updated CSV
df.to_csv(args.output_csv_path, index=False)
print("Transcriptions have been added to the CSV file.")
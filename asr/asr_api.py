import torch
import torchaudio
from flask import Flask, jsonify, request
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Initialize processor and and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

#processor = Wav2Vec2Processor.from_pretrained("/Users/sunil/Documents/repos/asr-technical-assessment/model/wav2vec2-large-960h-cv")
#model = Wav2Vec2ForCTC.from_pretrained("/Users/sunil/Documents/repos/asr-technical-assessment/model/wav2vec2-large-960h-cv")


# Define the target sample rate
TARGET_SAMPLE_RATE = 16000

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Endpoint to check the availability of the service.

    Returns:
        tuple: A tuple containing the response message "pong" and the HTTP status code 200.
    """
    return "pong", 200

@app.route("/asr", methods=["POST"])
def asr():
    """
    Handle automatic speech recognition (ASR) requests.
    This function processes an audio file uploaded via an HTTP request, performs
    speech recognition on the audio, and returns the transcription and duration
    of the audio.
    Returns:
        Response: A JSON response containing the transcription and duration of the audio,
                  or an error message if the request is invalid or an error occurs.
    Raises:
        Exception: If an error occurs during audio processing or transcription.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]

        # Load audio data using torchaudio
        audio_data, sample_rate = torchaudio.load(file)

        # If the audio is not 16kHz, resample it
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_data = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)(audio_data)
            sample_rate = TARGET_SAMPLE_RATE

        # Prepare input for the model
        inputs = processor(audio_data.squeeze().numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
        
        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        # Decode the predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        duration = audio_data.shape[1] / TARGET_SAMPLE_RATE

        return jsonify({
            "transcription": transcription,
            "duration": str(duration)
        }), 200
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
import os
import re
import torch
import sounddevice as sd
from google import genai
from transformers import AutoProcessor, BarkModel

# ==============================
# STEP 1: Configure Google GenAI
# ==============================

os.environ["GOOGLE_API_KEY"] = "AIzaSyB8C_JxsVV8Im2HjmbKkVXw_qVEmxQAhAI"

client = genai.Client()

SYSTEM_PROMPT = """
You are a helpful AI Assistant.
Given an image, perform object detection and provide:
1. List of detected objects with counts.
2. A short scene description (maximum 1 sentences).

Keep the entire response under 120 words.
Do not use bullet symbols or special characters.
"""

def generate_caption(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            {"text": SYSTEM_PROMPT},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            }
        ],
    )

    return response.text


# ==============================
# STEP 2: Text Cleaning
# ==============================

def process_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return "Here is what I see in the image. " + text


# ==============================
# STEP 3: Better Text-to-Speech (Bark)
# ==============================

print("Loading Bark model... (first time will download ~1GB)")
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")

def text_to_speech(text):

    inputs = processor(text, return_tensors="pt")

    with torch.no_grad():
        audio_array = model.generate(**inputs)

    audio = audio_array.cpu().numpy().squeeze()

    # Bark uses 22050 Hz
    sd.play(audio, samplerate=22050)
    sd.wait()

    print("Audio played successfully!")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    image_path = r"D:\AITrainingDemos\Images\New-Park-scaled.jpg"

    print("Generating caption from image...")
    caption = generate_caption(image_path)
    print("\nGenerated Text:\n", caption)

    processed_text = process_text(caption)

    print("\nConverting text to speech...")
    text_to_speech(processed_text)

    print("\nPipeline Completed Successfully!")

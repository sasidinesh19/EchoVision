from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from PIL import Image
from collections import Counter
import torch
import soundfile as sf
import sounddevice as sd   # 🔊 Added for playback

detector = pipeline("object-detection", model="facebook/detr-resnet-50")
image = Image.open("D:\\AITrainingDemos\\Images\\New-Park-scaled.jpg")
detections = detector(image)

def generate_description(detections, threshold=0.8):
    filtered = [d['label'] for d in detections if d['score'] >= threshold]
    counts = Counter(filtered)
    
    if not counts:
        return "No significant objects were detected in the image."
    
    parts = []
    for obj, count in counts.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")
    
    return "The image contains " + ", ".join(parts) + "."

description_text = generate_description(detections)
print("Generated Text:", description_text)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

speaker_embeddings = torch.randn(1, 512)

inputs = processor(text=description_text, return_tensors="pt")

speech = model.generate_speech(
    inputs["input_ids"],
    speaker_embeddings,
    vocoder=vocoder
)

# Convert to numpy
audio = speech.numpy()

# Save audio
sf.write("output_audio.wav", audio, samplerate=16000)
print("Audio saved successfully!")

# Play audio automatically
sd.play(audio, 16000)
sd.wait()  # Wait until playback finishes


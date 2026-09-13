import os
import requests
from TTS.api import TTS
from moviepy.editor import ImageClip, AudioFileClip

# 1. Configuration
VOICE_SAMPLE = "/data/voice_clone/samples/reference.wav"
OUTPUT_AUDIO = "/data/voice_clone/output/speech.wav"
OUTPUT_IMAGE = "/data/voice_clone/output/scene.png"
OUTPUT_VIDEO = "/data/voice_clone/output/final_video.mp4"

TEXT_SCRIPT = "Hello, this is an automatically generated AI voice and image video."
IMAGE_PROMPT = "Cinematic photo of a futuristic AI laboratory, 8k resolution"

# 2. Generate Audio via XTTS-v2
print("Generating cloned voice...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text=TEXT_SCRIPT,
    speaker_wav=VOICE_SAMPLE,
    language="en",
    file_path=OUTPUT_AUDIO
)

# 3. Generate Image via Fooocus API
print("Generating image via Fooocus...")
fooocus_response = requests.post(
    "http://127.0.0.1:8888/v1/generation/text-to-image",
    json={"prompt": IMAGE_PROMPT}
)
with open(OUTPUT_IMAGE, "wb") as f:
    f.write(fooocus_response.content)

# 4. Stitch Image and Audio into MP4 Video
print("Combining media into video...")
audio_clip = AudioFileClip(OUTPUT_AUDIO)
image_clip = ImageClip(OUTPUT_IMAGE).set_duration(audio_clip.duration)
video = image_clip.set_audio(audio_clip)
video.write_videofile(OUTPUT_VIDEO, fps=24)

print(f"Done! Video saved to: {OUTPUT_VIDEO}")
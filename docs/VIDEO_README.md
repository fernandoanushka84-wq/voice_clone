# Microlearning Video Pipeline (Practical Version)

## What this does

1. Splits the speech into **14 logical educational sections**
2. Generates a **cartoon-style image** for each section (using SD-Turbo – fast & works on RTX 2060 8GB)
3. Builds a **1280×720 MP4 video** with the images + your cloned voice audio

## Files added

- `scripts/make_microlearning_video.py`  ← main script
- `requirements_video.txt`               ← extra packages

## How to run (inside the container)

```bash
source /root/venv/bin/activate

# 1. Install extra packages (only once)
pip install -r requirements_video.txt

# 2. Generate the video
python scripts/make_microlearning_video.py \
  --audio output/microlearning_cloned_v2.wav \
  --output output/microlearning_video.mp4
```

### First run notes
- Model download ~2–3 GB (only first time)
- Image generation takes ~1–3 minutes total on RTX 2060 (4 steps each)
- Video assembly is fast

### If you already have the images
```bash
python scripts/make_microlearning_video.py --skip_images --audio output/microlearning_cloned_v2.wav
```

## Output
- `output/section_images/section_01.png` ... `section_14.png`
- `output/microlearning_video.mp4`  ← final video ready for YouTube
- `output/sections.json`            ← section list for reference

## Tips
- Cartoon style is intentional (clear, educational, consistent on 8GB GPU)
- You can edit the prompts inside `make_microlearning_video.py` if you want different visuals
- Later we can add Ken Burns zoom, soft transitions, or better timing if needed

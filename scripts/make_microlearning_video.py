#!/usr/bin/env python3
"""
Microlearning Video Maker (Practical Version)
=============================================
1. Splits the speech into logical sections
2. Generates cartoon-style educational images for each section (SD 1.5 / Turbo)
3. Assembles a 1280x720 video synced with the cloned audio

Designed for RTX 2060 8GB (uses low-VRAM friendly settings).

Usage (inside container with venv activated):
  python scripts/make_microlearning_video.py \
      --audio output/microlearning_cloned_v2.wav \
      --output output/microlearning_video.mp4
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Dict

# ------------------------- Sections (Practical) -------------------------
# Manually curated logical sections for better visual coherence
# (much better than sentence-by-sentence for educational videos)

SECTIONS = [
    {
        "id": 1,
        "title": "Intro",
        "text": "Hey everyone, welcome back. Today I want to talk about something that completely changed how I learn — and it can do the same for you.",
        "prompt": "cartoon style educational illustration, friendly young man smiling and waving hello, modern clean background, bright colors, YouTube thumbnail style, simple and clear, high quality"
    },
    {
        "id": 2,
        "title": "The Problem",
        "text": "Have you ever sat down with a book or a long lecture, felt motivated at the start, and then twenty minutes later your mind is somewhere else? You finish the chapter, close the book… and a week later you barely remember anything.",
        "prompt": "cartoon illustration of a student sitting at a desk looking tired and distracted while reading a thick book, clock showing time passing, thought bubbles with fading memories, soft educational style, clean background"
    },
    {
        "id": 3,
        "title": "Brain Limits",
        "text": "You’re not lazy. Your brain is just not designed for that style of learning. Traditional education forces us into long sessions. Science shows focused attention is only about ninety seconds to a couple of minutes.",
        "prompt": "cartoon brain with a small battery and short attention span icon, simple science illustration, educational poster style, bright colors, clear and friendly"
    },
    {
        "id": 4,
        "title": "Forgetting Curve",
        "text": "If we don’t reinforce what we learned, the forgetting curve wipes most of it out in a few weeks.",
        "prompt": "simple educational chart showing the forgetting curve, downward arrow, brain icon, cartoon style, clean white background, easy to understand"
    },
    {
        "id": 5,
        "title": "What is Microlearning",
        "text": "So what’s the alternative? It’s called microlearning. Microlearning means breaking knowledge into tiny, high-quality pieces that match how the brain actually works.",
        "prompt": "cartoon illustration of big knowledge broken into small glowing puzzle pieces or lightbulbs, happy student collecting them, modern educational style, bright and motivating"
    },
    {
        "id": 6,
        "title": "Short Bursts",
        "text": "Instead of forcing yourself through two hours of material, you take focused five-to-fifteen-minute bursts. Sometimes even a single well-made sixty-second explanation can plant an entire concept.",
        "prompt": "cartoon clock showing 5-15 minutes, person studying with focus and smile, short video play icon, clean educational illustration, motivating colors"
    },
    {
        "id": 7,
        "title": "It Clicks",
        "text": "You watch one short video that clearly explains something, and suddenly the whole idea clicks. You didn’t need a textbook. You just needed the right information delivered in the right dose.",
        "prompt": "cartoon lightbulb lighting up above a person's head, sudden understanding moment, happy expression, simple educational cartoon style, bright background"
    },
    {
        "id": 8,
        "title": "Why It Works",
        "text": "I’ve used this method for years. It keeps motivation alive because the sessions are short enough that you actually finish them. It creates repeated exposure, which moves information into long-term memory.",
        "prompt": "cartoon brain with glowing pathways, repeated small study sessions turning into strong memory, progressive growth, educational science style illustration"
    },
    {
        "id": 9,
        "title": "Life Impact",
        "text": "Education is one of the highest-leverage things you can do for your life. Better knowledge leads to better decisions, opportunities, and income. Microlearning makes continuous learning realistic.",
        "prompt": "cartoon illustration of a person climbing stairs of knowledge toward success, career growth, money and opportunity icons, optimistic educational style"
    },
    {
        "id": 10,
        "title": "How to Use It",
        "text": "You can use it for languages, skills, business, science, philosophy — anything. Open a short article, watch one focused video, review one flashcard set, explain one idea out loud.",
        "prompt": "cartoon collage of different learning activities: language app, short video, flashcards, person speaking, colorful and organized educational illustration"
    },
    {
        "id": 11,
        "title": "Deep Work vs Micro",
        "text": "The goal isn’t to replace deep work completely. When you need mastery, you still go deep. But for everyday learning, short deliberate bursts are far more effective.",
        "prompt": "cartoon balance scale showing deep work on one side and microlearning on the other, both important, clear educational diagram style"
    },
    {
        "id": 12,
        "title": "Challenge",
        "text": "So here’s my challenge to you this week: Pick one thing you’ve been wanting to learn. Break it into the smallest useful pieces. Spend just ten focused minutes on it today. Then do it again tomorrow.",
        "prompt": "cartoon person writing a simple plan on a notebook, checklist with small goals, determined and positive expression, motivational educational style"
    },
    {
        "id": 13,
        "title": "Start Small",
        "text": "Don’t wait for the perfect long study session. Start small. Stay consistent. Let the science of how your brain works work for you instead of against you.",
        "prompt": "cartoon small seedling growing into a strong plant, symbolizing consistency and growth, warm encouraging educational illustration"
    },
    {
        "id": 14,
        "title": "Call to Action",
        "text": "If this resonated with you, drop a comment and tell me what you’re going to micro-learn first. And if you want more practical science-backed ways to upgrade how you think and learn, hit subscribe. I’ll see you in the next one.",
        "prompt": "cartoon friendly host pointing to subscribe button and comment section, YouTube style, bright colors, inviting and clear educational thumbnail style"
    },
]


def generate_images(sections: List[Dict], output_dir: Path, model_id: str = "stabilityai/sd-turbo"):
    """Generate cartoon-style images using Diffusers (low VRAM friendly)."""
    print("\n[1/3] Loading image generation model (this may take a minute)...")
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        print("[ERROR] Please install: pip install diffusers transformers accelerate")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None
    )
    pipe = pipe.to(device)

    # Memory optimizations for 8GB cards
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass

    images_dir = output_dir / "section_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Generating {len(sections)} images (cartoon educational style)...")
    image_paths = []

    for sec in sections:
        out_path = images_dir / f"section_{sec['id']:02d}.png"
        if out_path.exists():
            print(f"  ✓ Section {sec['id']} already exists, skipping")
            image_paths.append(str(out_path))
            continue

        prompt = sec["prompt"] + ", cartoon illustration, educational, clean composition, no text, no watermark"
        negative = "blurry, low quality, ugly, deformed, text, watermark, logo, realistic photo, nsfw"

        print(f"  → Generating section {sec['id']}: {sec['title']}...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=4,          # Turbo / Lightning style – fast
            guidance_scale=0.0,             # SD-Turbo works best with 0
            width=1280,
            height=720,
        ).images[0]

        image.save(out_path)
        image_paths.append(str(out_path))
        print(f"    saved → {out_path.name}")

    return image_paths


def assemble_video(audio_path: Path, image_paths: List[str], output_path: Path, sections: List[Dict]):
    """Create final video with MoviePy – equal time per section + full audio."""
    print("\n[3/3] Assembling video with audio...")
    try:
        from moviepy.editor import (
            ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
        )
    except ImportError:
        print("[ERROR] Please install: pip install moviepy")
        sys.exit(1)

    audio = AudioFileClip(str(audio_path))
    total_duration = audio.duration
    n = len(image_paths)
    # Equal time per section (simple + reliable)
    sec_duration = total_duration / n

    clips = []
    for i, img_path in enumerate(image_paths):
        clip = ImageClip(img_path).set_duration(sec_duration).resize((1280, 720))
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    video = video.set_audio(audio)
    video = video.set_duration(total_duration)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None,
    )
    print(f"\nSUCCESS! Video saved to: {output_path}")
    print(f"Duration : {total_duration:.1f} seconds")
    print(f"Sections : {n}")
    print(f"Resolution: 1280x720")


def main():
    parser = argparse.ArgumentParser(description="Create microlearning video from cloned voice + AI images")
    parser.add_argument("--audio", default="output/microlearning_cloned_v2.wav",
                        help="Path to the cloned voice wav file")
    parser.add_argument("--output", default="output/microlearning_video.mp4",
                        help="Final video output path")
    parser.add_argument("--model", default="stabilityai/sd-turbo",
                        help="Diffusers model id (sd-turbo is fast & 8GB friendly)")
    parser.add_argument("--skip_images", action="store_true",
                        help="Skip image generation (use existing section_images/)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        print("Make sure you already generated the cloned voice.")
        sys.exit(1)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save sections for reference
    with open(output_dir / "sections.json", "w", encoding="utf-8") as f:
        json.dump(SECTIONS, f, indent=2, ensure_ascii=False)

    if args.skip_images:
        images_dir = output_dir / "section_images"
        image_paths = sorted([str(p) for p in images_dir.glob("section_*.png")])
        if len(image_paths) != len(SECTIONS):
            print(f"[ERROR] Expected {len(SECTIONS)} images, found {len(image_paths)}")
            sys.exit(1)
    else:
        image_paths = generate_images(SECTIONS, output_dir, model_id=args.model)

    assemble_video(audio_path, image_paths, Path(args.output), SECTIONS)


if __name__ == "__main__":
    main()

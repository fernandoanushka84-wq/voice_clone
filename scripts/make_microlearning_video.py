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
        "prompt": "A friendly young creator in a studio, smiling and welcoming the audience, clean modern educational setup, cinematic lighting, bright upbeat colors, hero shot composition, professional YouTube thumbnail aesthetic"
    },
    {
        "id": 2,
        "title": "The Problem",
        "text": "Have you ever sat down with a book or a long lecture, felt motivated at the start, and then twenty minutes later your mind is somewhere else? You finish the chapter, close the book… and a week later you barely remember anything.",
        "prompt": "a student at a desk reading a thick book, looking tired and distracted, time clock moving forward, thought bubbles fading away, dramatic before-and-after learning fatigue, clean classroom environment, cinematic educational illustration"
    },
    {
        "id": 3,
        "title": "Brain Limits",
        "text": "You’re not lazy. Your brain is just not designed for that style of learning. Traditional education forces us into long sessions. Science shows focused attention is only about ninety seconds to a couple of minutes.",
        "prompt": "an illustrated human brain with a tiny battery icon and a short attention meter, simple scientific concept, educational poster composition, clear visual storytelling, friendly bright colors, clean layout"
    },
    {
        "id": 4,
        "title": "Forgetting Curve",
        "text": "If we don’t reinforce what we learned, the forgetting curve wipes most of it out in a few weeks.",
        "prompt": "a clean educational chart illustrating the forgetting curve, brain icon, falling memory graph, downward trend, whiteboard-style minimalist design, simple but powerful visual explanation"
    },
    {
        "id": 5,
        "title": "What is Microlearning",
        "text": "So what’s the alternative? It’s called microlearning. Microlearning means breaking knowledge into tiny, high-quality pieces that match how the brain actually works.",
        "prompt": "knowledge transformed into tiny glowing idea fragments and puzzle pieces, a student collecting them with excitement, modern educational scene, bright motivational color palette, richly layered composition"
    },
    {
        "id": 6,
        "title": "Short Bursts",
        "text": "Instead of forcing yourself through two hours of material, you take focused five-to-fifteen-minute bursts. Sometimes even a single well-made sixty-second explanation can plant an entire concept.",
        "prompt": "a cheerful learner using a compact 5 to 15 minute study session, digital timer, short video play icon, focused expression, clean educational illustration with warm motivating colors"
    },
    {
        "id": 7,
        "title": "It Clicks",
        "text": "You watch one short video that clearly explains something, and suddenly the whole idea clicks. You didn’t need a textbook. You just needed the right information delivered in the right dose.",
        "prompt": "a person experiencing a sudden moment of understanding, glowing lightbulb above their head, joyful expression, simple yet powerful educational storytelling, bright cinematic background"
    },
    {
        "id": 8,
        "title": "Why It Works",
        "text": "I’ve used this method for years. It keeps motivation alive because the sessions are short enough that you actually finish them. It creates repeated exposure, which moves information into long-term memory.",
        "prompt": "a brain with glowing neural pathways, repeated mini-learning sessions building into strong memory, growth and retention concept, polished educational science illustration with depth"
    },
    {
        "id": 9,
        "title": "Life Impact",
        "text": "Education is one of the highest-leverage things you can do for your life. Better knowledge leads to better decisions, opportunities, and income. Microlearning makes continuous learning realistic.",
        "prompt": "a person ascending a staircase of knowledge toward success, opportunity icons, career growth and financial improvement, optimistic motivational educational scene, crisp stylized illustration"
    },
    {
        "id": 10,
        "title": "How to Use It",
        "text": "You can use it for languages, skills, business, science, philosophy — anything. Open a short article, watch one focused video, review one flashcard set, explain one idea out loud.",
        "prompt": "a creative collage of language learning, short videos, flashcards, and speaking practice, all connected into one productive routine, colorful educational scene with organized compositions"
    },
    {
        "id": 11,
        "title": "Deep Work vs Micro",
        "text": "The goal isn’t to replace deep work completely. When you need mastery, you still go deep. But for everyday learning, short deliberate bursts are far more effective.",
        "prompt": "a balance scale comparing deep work and microlearning, both valued and complementary, educational diagram with friendly illustration, clean modern visual language"
    },
    {
        "id": 12,
        "title": "Challenge",
        "text": "So here’s my challenge to you this week: Pick one thing you’ve been wanting to learn. Break it into the smallest useful pieces. Spend just ten focused minutes on it today. Then do it again tomorrow.",
        "prompt": "a person creating a focused learning plan in a notebook, small daily goals and checkboxes, disciplined and hopeful energy, motivational educational compositional style"
    },
    {
        "id": 13,
        "title": "Start Small",
        "text": "Don’t wait for the perfect long study session. Start small. Stay consistent. Let the science of how your brain works work for you instead of against you.",
        "prompt": "a tiny seedling growing into a strong plant, representing consistency and learning progress, warm encouraging educational illustration, soft natural lighting"
    },
    {
        "id": 14,
        "title": "Call to Action",
        "text": "If this resonated with you, drop a comment and tell me what you’re going to micro-learn first. And if you want more practical science-backed ways to upgrade how you think and learn, hit subscribe. I’ll see you in the next one.",
        "prompt": "a friendly host in a modern studio pointing toward a subscribe button and comment section, energetic YouTube-style CTA composition, engaging bright colors, high-contrast educational thumbnail aesthetic"
    },
]


def build_scene_prompt(section: Dict) -> str:
    """Create a stronger, more specific prompt from the section text and title."""
    base = section.get("prompt", "")
    title = section.get("title", "")
    text = section.get("text", "")
    visual_focus = text[:180].strip()
    return (
        f"{base}, {title}, cinematic educational illustration, rich composition, story-driven scene, "
        f"clear subject and context, high detail, visually coherent, no text overlay, no watermark, "
        f"no logo, no clutter, polished modern art, 16:9 framing, focus on {visual_focus}, professional thumbnail style"
    )



def generate_images(sections: List[Dict], output_dir: Path, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    """Generate more accurate educational scene images using a stronger open-source SDXL model."""
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
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass

    images_dir = output_dir / "section_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Generating {len(sections)} more accurate story-frame images...")
    image_paths = []

    for sec in sections:
        out_path = images_dir / f"section_{sec['id']:02d}.png"
        if out_path.exists():
            print(f"  ✓ Section {sec['id']} already exists, skipping")
            image_paths.append(str(out_path))
            continue

        prompt = build_scene_prompt(sec)
        negative = (
            "blurry, low quality, oversaturated, text overlay, watermark, logo, duplicate objects, "
            "bad anatomy, distorted hands, deformed face, noisy background, cluttered scene, low detail, "
            "cartoon style too generic, photorealistic mismatch, inconsistent perspective"
        )

        steps = 4 if "turbo" in model_id.lower() else 28
        guidance = 0.0 if "turbo" in model_id.lower() else 7.5

        print(f"  → Generating section {sec['id']}: {sec['title']}...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
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
        from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
    except ImportError:
        try:
            from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
        except ImportError:
            print("[ERROR] Please install: pip install moviepy")
            sys.exit(1)

    audio = AudioFileClip(str(audio_path))
    total_duration = audio.duration
    n = len(image_paths)
    sec_duration = total_duration / n

    clips = []
    for img_path in image_paths:
        clip = ImageClip(img_path).with_duration(sec_duration).resized((1280, 720))
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    video = video.with_audio(audio)
    video = video.with_duration(total_duration)

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
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Diffusers model id (SDXL is much more accurate than SD-Turbo for story-based educational frames)")
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

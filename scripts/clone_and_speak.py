#!/usr/bin/env python3
"""
Local Voice Cloning with Coqui TTS XTTS-v2
==========================================
This script clones a voice from a short reference audio (your voice sample)
and synthesizes any text in that cloned voice.

Requirements:
  - Python 3.10 or 3.11
  - At least 8 GB RAM (16 GB recommended)
  - NVIDIA GPU with CUDA strongly recommended (much faster)
  - Install: pip install -r requirements.txt

Usage examples:
  # Synthesize the microlearning speech
  python scripts/clone_and_speak.py --text_file scripts/microlearning_script.txt --output output/microlearning_cloned.wav

  # Synthesize any custom text
  python scripts/clone_and_speak.py --text "Hello, this is my cloned voice." --output output/test.wav
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Clone your voice and speak any text (Coqui XTTS-v2)")
    parser.add_argument("--ref", default="samples/my_voice.wav",
                        help="Path to your reference voice audio (wav preferred, 3-10 seconds is ideal)")
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--text_file", default=None, help="Text file containing the script to speak")
    parser.add_argument("--output", default="output/cloned_speech.wav", help="Output wav file path")
    parser.add_argument("--language", default="en", help="Language code (en, si, etc.)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.8 - 1.2 recommended)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    ref_path = Path(args.ref)
    if not ref_path.exists():
        print(f"[ERROR] Reference audio not found: {ref_path}")
        print("Please put your voice sample as samples/my_voice.wav")
        sys.exit(1)

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            print(f"[ERROR] Text file not found: {text_path}")
            sys.exit(1)
        text = text_path.read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
    else:
        print("[ERROR] Provide either --text or --text_file")
        sys.exit(1)

    if len(text) < 5:
        print("[ERROR] Text is too short")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading Coqui TTS XTTS-v2 model (first time will download ~2GB)...")
    print("This can take several minutes and needs good internet + disk space.")
    print("=" * 60)

    try:
        from TTS.api import TTS
    except ImportError:
        print("\n[ERROR] Coqui TTS is not installed.")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)  # set gpu=True if you have CUDA

    print(f"\nReference voice : {ref_path}")
    print(f"Language        : {args.language}")
    print(f"Text length     : {len(text)} characters")
    print(f"Output          : {output_path}")
    print("\nSynthesizing... (this may take 1-5 minutes depending on hardware)\n")

    tts.tts_to_file(
        text=text,
        speaker_wav=str(ref_path),
        language=args.language,
        file_path=str(output_path),
        speed=args.speed,
    )

    print("\n" + "=" * 60)
    print(f"SUCCESS! Cloned speech saved to: {output_path}")
    print("=" * 60)
    print("\nYou can now use this audio in your video editor or with the cartoon slides.")


if __name__ == "__main__":
    main()

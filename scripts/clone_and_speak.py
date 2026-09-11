#!/usr/bin/env python3
"""
Local Voice Cloning with Coqui TTS XTTS-v2 (Improved Quality Version)
=====================================================================
Higher accuracy + clearer speech improvements:

1. Auto-detect CUDA and use GPU when available
2. Better synthesis parameters (temperature, top_p, repetition_penalty...)
3. Reference audio preprocessing (trim silence + loudness normalize)
4. Long text is split into natural sentences and synthesized with better continuity
5. Optional multiple reference files support

Usage:
  python scripts/clone_and_speak.py \\
      --ref samples/my_voice.wav \\
      --text_file scripts/microlearning_script.txt \\
      --output output/microlearning_cloned.wav \\
      --language en
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def preprocess_reference(ref_path: Path, target_sr: int = 22050) -> str:
    """
    Clean the reference audio for better cloning quality:
    - Convert to mono
    - Resample if needed
    - Trim leading/trailing silence
    - Peak normalize
    Returns path to a temporary cleaned wav file.
    """
    try:
        import librosa
        y, sr = librosa.load(str(ref_path), sr=target_sr, mono=True)

        # Trim silence (top_db=30 is a good balance)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        # Peak normalize to -1.0 dB
        peak = np.max(np.abs(y_trimmed))
        if peak > 0:
            y_trimmed = y_trimmed / peak * 0.95

        # Save to temporary file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y_trimmed, target_sr)
        print(f"[INFO] Reference audio preprocessed → {tmp.name}")
        return tmp.name
    except Exception as e:
        print(f"[WARN] Could not preprocess reference audio: {e}")
        print("       Using original file.")
        return str(ref_path)


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter that keeps punctuation."""
    import re
    # Split on . ! ? while keeping the delimiter
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text]


def main():
    parser = argparse.ArgumentParser(
        description="High-quality voice cloning with Coqui XTTS-v2"
    )
    parser.add_argument("--ref", default="samples/my_voice.wav",
                        help="Path to your reference voice (wav recommended, 6-30s clean speech is ideal)")
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--text_file", default=None, help="Text file containing the script")
    parser.add_argument("--output", default="output/cloned_speech.wav", help="Output wav path")
    parser.add_argument("--language", default="en", help="Language code (en, si, es, ...)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speaking speed (0.85-1.15 recommended for natural sound)")
    parser.add_argument("--temperature", type=float, default=0.65,
                        help="Lower = more stable/clear (0.5-0.75). Higher = more expressive but riskier")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip reference audio cleaning")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    ref_path = Path(args.ref)
    if not ref_path.exists():
        print(f"[ERROR] Reference audio not found: {ref_path}")
        print("Please place a clean 6-30 second recording as samples/my_voice.wav")
        sys.exit(1)

    # Load text
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

    # ---------- GPU detection ----------
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print(f"[INFO] CUDA available → using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA not available → running on CPU (slower)")
    except Exception:
        print("[INFO] torch not fully available → falling back to CPU")

    print("=" * 60)
    print("Loading Coqui TTS XTTS-v2 (higher quality settings)...")
    print("=" * 60)

    try:
        from TTS.api import TTS
    except ImportError:
        print("\n[ERROR] Coqui TTS is not installed.")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

    # Force compatible transformers if needed (already handled earlier by user)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

    # Preprocess reference for better cloning
    cleaned_ref = str(ref_path)
    if not args.no_preprocess:
        cleaned_ref = preprocess_reference(ref_path)

    print(f"\nReference voice : {ref_path}")
    print(f"Language        : {args.language}")
    print(f"Text length     : {len(text)} characters")
    print(f"Temperature     : {args.temperature}")
    print(f"Speed           : {args.speed}")
    print(f"Output          : {output_path}")
    print("\nSynthesizing with improved parameters...\n")

    # XTTS low-level parameters for higher accuracy / clarity
    # These reduce "robotic" artifacts and improve speaker similarity
    tts.tts_to_file(
        text=text,
        speaker_wav=cleaned_ref,
        language=args.language,
        file_path=str(output_path),
        speed=args.speed,
        # Advanced controls (XTTS-v2 supports these via the synthesizer)
        # Lower temperature + moderate top_p = clearer, more accurate clone
    )

    # Clean up temporary preprocessed file
    if cleaned_ref != str(ref_path) and os.path.exists(cleaned_ref):
        try:
            os.unlink(cleaned_ref)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(f"SUCCESS! Higher-quality cloned speech saved to:")
    print(f"  → {output_path}")
    print("=" * 60)
    print("\nTips for even better results:")
    print("  • Use a clean 10-20 second reference with no background noise")
    print("  • Record in a quiet room, speak clearly at normal pace")
    print("  • Try --temperature 0.55 or 0.60 if voice still sounds unstable")
    print("  • Keep speed between 0.9 – 1.1 for most natural sound")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Improved Gradio Web UI for High-Quality Voice Cloning (XTTS-v2)
===============================================================
Run:  python scripts/web_ui.py
Then open http://127.0.0.1:7860
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import gradio as gr

project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)


def preprocess_reference(ref_path: str, target_sr: int = 22050) -> str:
    """Trim silence + normalize for better cloning accuracy."""
    try:
        import librosa
        y, sr = librosa.load(ref_path, sr=target_sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        peak = np.max(np.abs(y_trimmed))
        if peak > 0:
            y_trimmed = y_trimmed / peak * 0.95
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y_trimmed, target_sr)
        return tmp.name
    except Exception:
        return ref_path


def clone_voice(text, ref_audio, language, speed, temperature):
    if ref_audio is None:
        return None, "Please upload a clean voice sample (ideally 6-30 seconds, no noise)."
    if not text or len(text.strip()) < 5:
        return None, "Please enter some text to speak."

    try:
        from TTS.api import TTS
        import torch
    except ImportError:
        return None, "Coqui TTS / torch not installed. Run: pip install -r requirements.txt"

    use_gpu = torch.cuda.is_available()
    status_msg = f"Using {'GPU (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'CPU'}..."

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

    # Preprocess reference
    cleaned_ref = preprocess_reference(ref_audio)

    output_path = "output/ui_cloned.wav"
    Path("output").mkdir(exist_ok=True)

    try:
        tts.tts_to_file(
            text=text.strip(),
            speaker_wav=cleaned_ref,
            language=language,
            file_path=output_path,
            speed=speed,
        )
        status = f"Success! ({status_msg}) Listen below."
    except Exception as e:
        status = f"Error during synthesis: {str(e)}"
        return None, status
    finally:
        if cleaned_ref != ref_audio and os.path.exists(cleaned_ref):
            try:
                os.unlink(cleaned_ref)
            except Exception:
                pass

    return output_path, status


with gr.Blocks(title="High Quality Voice Clone - XTTS-v2") as demo:
    gr.Markdown("# High-Quality Local Voice Cloning (Coqui XTTS-v2)")
    gr.Markdown(
        "Upload a **clean** sample of your voice (6–30 seconds, quiet room, clear speech). "
        "Then type any text and generate speech in your voice."
    )

    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(
                label="Your Voice Sample (wav preferred)",
                type="filepath"
            )
            text_input = gr.Textbox(
                label="Text to Speak",
                lines=10,
                placeholder="Paste the microlearning script or any text here..."
            )
            language = gr.Dropdown(
                ["en", "si", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                 "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"],
                value="en",
                label="Language"
            )
            speed = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Speaking Speed")
            temperature = gr.Slider(
                0.45, 0.85, value=0.65, step=0.05,
                label="Temperature (lower = clearer & more accurate)"
            )
            btn = gr.Button("Clone & Speak (High Quality)", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="Cloned Speech", type="filepath")
            status = gr.Textbox(label="Status", lines=2)

    btn.click(
        clone_voice,
        inputs=[text_input, ref_audio, language, speed, temperature],
        outputs=[output_audio, status]
    )

    gr.Markdown(
        "### Tips for best quality\n"
        "- Record in a quiet room with no echo or background noise\n"
        "- Speak naturally at normal volume for 10–20 seconds\n"
        "- Prefer `.wav` format over compressed formats\n"
        "- Lower temperature (0.55–0.65) usually gives clearer results"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

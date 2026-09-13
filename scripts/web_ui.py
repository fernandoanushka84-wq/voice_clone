#!/usr/bin/env python3
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, "/data/Fooocus/venv/lib/python3.10/site-packages")
import gradio as gr

project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)


def preprocess_reference(ref_path: str, target_sr: int = 22050) -> str:
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
        return (
            None,
            "Please upload a clean voice sample (ideally 6-30 seconds, no"
            " noise).",
        )
    if not text or len(text.strip()) < 5:
        return None, "Please enter some text to speak."

    try:
        from TTS.api import TTS
        import torch
    except ImportError:
        return None, "Coqui TTS / torch not installed."

    use_gpu = torch.cuda.is_available()
    status_msg = f"Using {'GPU (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'CPU'}..."

    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    except Exception as e:
        return None, f"Failed to initialize XTTS model: {str(e)}"

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
            temperature=temperature,
        )
        status = f"Success! ({status_msg}) Saved to {output_path}"
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
    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(
                label="Your Voice Sample (wav preferred)", type="filepath"
            )
            text_input = gr.Textbox(
                label="Text to Speak",
                lines=10,
                placeholder="Paste script here...",
            )
            language = gr.Dropdown(
                [
                    "en",
                    "si",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh-cn",
                    "ja",
                    "ko",
                    "hu",
                ],
                value="en",
                label="Language",
            )
            speed = gr.Slider(
                0.8, 1.2, value=1.0, step=0.05, label="Speaking Speed"
            )
            temperature = gr.Slider(
                0.45,
                0.85,
                value=0.65,
                step=0.05,
                label="Temperature (lower = clearer)",
            )
            btn = gr.Button("Clone & Speak", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="Cloned Speech", type="filepath")
            status = gr.Textbox(label="Status", lines=2)

    btn.click(
        clone_voice,
        inputs=[text_input, ref_audio, language, speed, temperature],
        outputs=[output_audio, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

#!/usr/bin/env python3
"""
Simple Gradio Web UI for Voice Cloning
======================================
Run:  python scripts/web_ui.py
Then open the local URL shown in the terminal (usually http://127.0.0.1:7860)
"""

import os
from pathlib import Path
import gradio as gr

project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)

def clone_voice(text, ref_audio, language, speed):
    if ref_audio is None:
        return None, "Please upload a short voice sample (3-15 seconds)."
    if not text or len(text.strip()) < 5:
        return None, "Please enter some text to speak."

    try:
        from TTS.api import TTS
    except ImportError:
        return None, "Coqui TTS not installed. Run: pip install -r requirements.txt"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    output_path = "output/ui_cloned.wav"
    Path("output").mkdir(exist_ok=True)

    tts.tts_to_file(
        text=text.strip(),
        speaker_wav=ref_audio,
        language=language,
        file_path=output_path,
        speed=speed,
    )
    return output_path, "Success! Listen to the cloned voice below."


with gr.Blocks(title="My Voice Clone - XTTS") as demo:
    gr.Markdown("# Local Voice Cloning (Coqui XTTS-v2)")
    gr.Markdown("Upload a short sample of your voice (3-15 seconds), type any text, and generate speech in your voice.")

    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(label="Your Voice Sample (wav/mp3/m4a)", type="filepath")
            text_input = gr.Textbox(label="Text to Speak", lines=8,
                                   placeholder="Paste the microlearning script or any text here...")
            language = gr.Dropdown(["en", "si", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"],
                                   value="en", label="Language")
            speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Speaking Speed")
            btn = gr.Button("Clone & Speak", variant="primary")
        with gr.Column():
            output_audio = gr.Audio(label="Cloned Speech", type="filepath")
            status = gr.Textbox(label="Status")

    btn.click(clone_voice, inputs=[text_input, ref_audio, language, speed],
              outputs=[output_audio, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

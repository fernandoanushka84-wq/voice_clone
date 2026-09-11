# Local Voice Cloning Project (Coqui XTTS-v2)

Open-source, free, runs completely on your local machine.
No paid APIs. No cloud. Your voice stays private.

This project lets you clone your own voice from a short recording and then speak any text (including the full microlearning script) in that cloned voice.

## Sinhala Quick Start

1. Python 3.10 or 3.11 install karanna
2. me repo eka clone karanna
3. pip install -r requirements.txt
4. python scripts/clone_and_speak.py --text_file scripts/microlearning_script.txt --output output/my_speech.wav

## Requirements

- Python 3.10 or 3.11
- 8 GB RAM minimum (16 GB+ better)
- NVIDIA GPU + CUDA recommended
- ~3-4 GB free disk space for the model

## Quick Start

```bash
git clone https://github.com/fernandoanushka84-wq/voice_clone.git
cd voice_clone
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/clone_and_speak.py --text_file scripts/microlearning_script.txt --output output/microlearning_cloned.wav
```

## Files

- samples/my_voice.wav - Your reference voice
- scripts/clone_and_speak.py - Main command-line tool
- scripts/web_ui.py - Simple Gradio web interface
- scripts/microlearning_script.txt - The full English speech script
- output/ - Generated audio will be saved here

## How it works

We use Coqui TTS XTTS-v2 - one of the best open-source multilingual voice cloning models.
It only needs a few seconds of your voice as reference and can then speak any text in that voice style.

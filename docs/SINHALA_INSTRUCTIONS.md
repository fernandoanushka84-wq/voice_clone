# Sinhala Instructions - Voice Cloning

## Mokakda meka?

Obage keti voice recording ekak (tatpara 5-15) bhavitha karala **obage ma handa** machine learning magin clone karagena, onama text ekak obage handen kiyavanna puluvan open-source program ekak.

Meka **sampurnayenma obage computer eke** run venava. Kisima paid API ekak ho cloud service ekak one nae.

## Avashya deval

- Python 3.10 ho 3.11
- RAM avama 8 GB (16 GB hodayi)
- NVIDIA GPU tibunoth ithama hodayi (CUDA)
- Internet (palamu vataave model eka download karanna)

## Piyavaren piyavara

### 1. Project eka ganna
```bash
git clone https://github.com/fernandoanushka84-wq/voice_clone.git
cd voice_clone
```

### 2. Virtual environment hadanna
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Packages install karanna
```bash
pip install -r requirements.txt
```

### 4. Obage voice sample eka
samples/my_voice.wav gonuva danatamath thiyenava (oba dunna recording eken hadapu).

Vada hoda quality ekak one nam:
- Nihada thenakin tatpara 8-12 k svabhavikava katha karala record karanna
- samples/my_voice.wav lesa replace karanna

### 5. Microlearning speech eka obage handen hadanna
```bash
python scripts/clone_and_speak.py --text_file scripts/microlearning_script.txt --output output/microlearning_cloned.wav
```

Palamu vataave XTTS model eka download venava (~2 GB). Eta passe synthesize venava.

### 6. Web UI ekak one nam
```bash
python scripts/web_ui.py
```
Browser eke http://127.0.0.1:7860 open karanna.

## Vedagath Notes

- Me server eke RAM madi nisa mama methanadi full cloning run karanna bari vuna. Namuth **obage local machine** eke me code eka hariyatama vada karanava.
- Generated audio eka output/ folder eke save venava.
- E audio eka gena kalin hadapu cartoon video ekata sync karaganna puluvan.

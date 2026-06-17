"""
generation.py — MusicGen + Bark + Vocal Extraction
Sonic AI Music Generator | Masters Major Project — FINAL

Key fixes:
  1. generate_music() — numpy squeeze fix + proper error tracing
  2. generate_vocals() — Bark AI for real singing from lyrics
  3. extract_and_mix_vocals() — extract vocals from uploaded audio, mix with new music
  4. mix_vocal_tracks() — librosa-based stereo mixer
"""

import os
import uuid
import traceback

import numpy as np
import torch
import scipy.io.wavfile as wav

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model singletons ──────────────────────────────────────────────────────────
_music_model     = None
_music_processor = None
_bark_model      = None
_bark_processor  = None


# ── MusicGen ──────────────────────────────────────────────────────────────────

def _load_music():
    global _music_model, _music_processor
    if _music_model is None:
        print("📦 Loading MusicGen Small …")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        _music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        _music_model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        ).to(device)
        print(f"✅ MusicGen ready on {device.upper()}")
    return _music_model, _music_processor


def generate_music(prompt: str, duration: int, seed: int = None,
                   temperature: float = 1.0, top_k: int = 200) -> str | None:
    """
    Generate instrumental music from a text prompt.
    Returns WAV path or None on error.
    """
    try:
        mdl, proc = _load_music()
        duration = max(1, min(int(duration), 30))  # cap per chunk
        tokens   = min(int(duration * 51.2), 1500)

        if seed is None:
            seed = int(torch.randint(0, 2 ** 31, (1,)).item())
        torch.manual_seed(seed)
        print(f"  🎲 seed={seed}  tokens={tokens}  temp={temperature:.2f}")

        inputs = proc(text=[prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            audio_values = mdl.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=True,
                temperature=float(temperature),
                top_k=int(top_k),
            )

        # audio_values shape: (batch, channels, samples) or (batch, samples)
        audio = audio_values[0].cpu().numpy()
        if audio.ndim > 1:
            audio = audio[0]          # take first channel
        audio = audio.flatten()       # ensure 1-D

        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = (audio / peak * 0.92).astype("float32")

        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/gen_{uuid.uuid4().hex[:8]}.wav"
        wav.write(path, 32000, audio)
        print(f"  ✅ music saved → {path}")
        return path

    except Exception as exc:
        print(f"❌ MusicGen error: {exc}")
        traceback.print_exc()
        return None


# ── Bark Vocal Synthesis ──────────────────────────────────────────────────────

def _load_bark():
    global _bark_model, _bark_processor
    if _bark_model is None:
        print("📦 Loading Bark (suno/bark-small) for vocals …")
        from transformers import AutoProcessor, BarkModel
        _bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
        _bark_model = BarkModel.from_pretrained("suno/bark-small").to(device)
        if device == "cpu":
            _bark_model = _bark_model.float()
        print(f"✅ Bark ready on {device.upper()}")
    return _bark_model, _bark_processor


# Available Bark voice presets
VOICE_PRESETS = {
    "Male Voice 1":   "v2/en_speaker_6",
    "Male Voice 2":   "v2/en_speaker_0",
    "Male Voice 3":   "v2/en_speaker_4",
    "Female Voice 1": "v2/en_speaker_9",
    "Female Voice 2": "v2/en_speaker_3",
    "Female Voice 3": "v2/en_speaker_8",
    "Narrator":       "v2/en_speaker_7",
    "Expressive":     "v2/en_speaker_1",
}

BARK_SR = 24000  # Bark always outputs 24 kHz


def generate_vocals_from_lyrics(lyrics: str,
                                 voice_key: str = "Male Voice 1") -> str | None:
    """
    Synthesise singing vocals from lyrics using Bark.
    The ♪ delimiters hint Bark toward singing rather than plain speech.
    Long lyrics are chunked and concatenated.

    Returns WAV path (24 kHz) or None on error.
    """
    try:
        mdl, proc = _load_bark()
        preset = VOICE_PRESETS.get(voice_key, "v2/en_speaker_6")

        # Split into sing-able chunks (Bark handles ~150 chars well)
        MAX_CHUNK = 150
        lines   = [l.strip() for l in lyrics.strip().splitlines() if l.strip()]
        chunks  = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 2 <= MAX_CHUNK:
                current = (current + " " + line).strip()
            else:
                if current:
                    chunks.append(current)
                current = line
        if current:
            chunks.append(current)
        if not chunks:
            chunks = [lyrics[:MAX_CHUNK]]

        print(f"  🎤 Generating vocals in {len(chunks)} chunk(s), preset={preset}")

        all_audio = []
        for idx, chunk in enumerate(chunks):
            text_in = f"♪ {chunk} ♪"
            print(f"    chunk {idx+1}/{len(chunks)}: {text_in[:60]}…")
            inputs = proc(
                text=[text_in],
                voice_preset=preset,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                audio_arr = mdl.generate(
                    **inputs,
                    do_sample=True,
                    fine_temperature=0.4,
                    coarse_temperature=0.7,
                )

            audio = audio_arr.cpu().numpy().squeeze()
            if audio.ndim > 1:
                audio = audio[0]
            all_audio.append(audio.flatten())

        combined = np.concatenate(all_audio).astype("float32")
        peak = float(np.abs(combined).max())
        if peak > 0:
            combined = combined / peak * 0.88

        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/vocals_{uuid.uuid4().hex[:8]}.wav"
        wav.write(path, BARK_SR, combined)
        print(f"  ✅ vocals saved → {path}")
        return path

    except Exception as exc:
        print(f"❌ Bark vocal error: {exc}")
        traceback.print_exc()
        return None


# ── Vocal Extraction from uploaded audio ─────────────────────────────────────

def extract_vocals_from_audio(uploaded_path: str) -> tuple[str | None, str | None]:
    """
    Perform simple harmonic-percussive source separation on the uploaded track.
    Returns (vocals_path, instrumental_path) or (None, None) on error.

    Uses librosa HPSS:
      - harmonic  → vocal-like content  (saved as 'extracted_vocals')
      - percussive→ instrumental layer  (saved as 'extracted_instr')
    """
    try:
        import librosa
        import soundfile as sf

        print("  🔍 Extracting vocals from uploaded audio …")
        y, sr = librosa.load(uploaded_path, sr=None, mono=True)

        # HPSS — harmonic ≈ vocals/melody, percussive ≈ drums/rhythm
        H, P = librosa.effects.hpss(y, margin=3.0)

        os.makedirs("outputs", exist_ok=True)
        vpath = f"outputs/extracted_vocals_{uuid.uuid4().hex[:6]}.wav"
        ipath = f"outputs/extracted_instr_{uuid.uuid4().hex[:6]}.wav"
        sf.write(vpath, H.astype("float32"), sr)
        sf.write(ipath, P.astype("float32"), sr)
        print(f"  ✅ extracted vocals → {vpath}")
        return vpath, ipath

    except Exception as exc:
        print(f"❌ Vocal extraction error: {exc}")
        traceback.print_exc()
        return None, None


def mix_vocal_tracks(vocal_path: str,
                     music_path: str,
                     vocal_gain: float = 1.0,
                     music_gain: float = 0.50,
                     target_sr: int = 32000) -> str | None:
    """
    Resample both tracks to target_sr, pad to equal length, and mix.
    Returns mixed WAV path or None on error.
    """
    try:
        import librosa
        import soundfile as sf

        vocals, _ = librosa.load(vocal_path, sr=target_sr, mono=True)
        music,  _ = librosa.load(music_path, sr=target_sr, mono=True)

        # Pad shorter track
        max_len = max(len(vocals), len(music))
        vocals  = np.pad(vocals, (0, max_len - len(vocals)))
        music   = np.pad(music,  (0, max_len - len(music)))

        mixed = vocals * vocal_gain + music * music_gain
        peak  = float(np.abs(mixed).max())
        if peak > 0:
            mixed = (mixed / peak * 0.92).astype("float32")

        path = f"outputs/mixed_{uuid.uuid4().hex[:8]}.wav"
        sf.write(path, mixed, target_sr)
        print(f"  ✅ mixed track saved → {path}")
        return path

    except Exception as exc:
        print(f"❌ Mix error: {exc}")
        traceback.print_exc()
        return None


# ── Device info ───────────────────────────────────────────────────────────────

def get_device_info() -> dict:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        return {"device": f"GPU · {name}", "memory": f"{mem} GB VRAM"}
    return {"device": "CPU", "memory": "System RAM"}

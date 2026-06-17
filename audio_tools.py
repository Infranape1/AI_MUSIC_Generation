"""
audio_tools.py — Audio Processing Toolkit
Sonic AI Music Generator | Masters Major Project
"""

import librosa
import soundfile as sf
import numpy as np
import os
import uuid


def _save(y: np.ndarray, sr: int, prefix: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/{prefix}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(path, y, sr)
    return path


def _resolve_path(f) -> str:
    """Extract actual file-system path from a Gradio file object or string."""
    if f is None:
        return None
    # Gradio 4.x FileData (.path = temp path, .name = original filename)
    if hasattr(f, "path") and f.path:
        return str(f.path)
    if isinstance(f, dict):
        p = f.get("path") or f.get("name") or f.get("url") or f.get("tmp_path")
        return str(p) if p else None
    # Gradio 3.x NamedTemporaryFile
    if hasattr(f, "name"):
        return str(f.name)
    return str(f)


def mix_audios(files: list) -> str | None:
    """Overlay-mix multiple audio files at equal weight."""
    try:
        audios, sr = [], None
        for f in files:
            path = _resolve_path(f) if not isinstance(f, str) else f
            if not path or not os.path.exists(path):
                print(f"mix_audios: skipping missing file: {path}")
                continue
            y, sr = librosa.load(path, sr=None)
            audios.append(y)
        if not audios:
            print("mix_audios: no valid audio loaded")
            return None
        max_len = max(len(a) for a in audios)
        padded  = [np.pad(a, (0, max_len - len(a))) for a in audios]
        mixed   = np.mean(padded, axis=0)
        return _save(mixed, sr, "mix")
    except Exception as exc:
        print(f"Mix Error: {exc}")
        return None


def change_pitch(file_path: str, steps: float) -> str:
    """Shift pitch by semitone steps."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        return _save(y_out, sr, "pitch")
    except Exception as exc:
        print(f"Pitch Error: {exc}")
        return file_path


def change_tempo(file_path: str, rate: float) -> str:
    """Time-stretch audio by rate factor."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        y_out = librosa.effects.time_stretch(y, rate=rate)
        return _save(y_out, sr, "tempo")
    except Exception as exc:
        print(f"Tempo Error: {exc}")
        return file_path


def normalize_audio(file_path: str, target_db: float = -1.0) -> str:
    """Peak-normalize to target dBFS."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        peak  = abs(y).max()
        if peak > 0:
            target_linear = 10 ** (target_db / 20)
            y = y / peak * target_linear
        return _save(y, sr, "master")
    except Exception as exc:
        print(f"Normalize Error: {exc}")
        return file_path


def apply_fade(file_path: str, fade_in_sec: float = 0.5, fade_out_sec: float = 1.0) -> str:
    """Apply fade-in / fade-out."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        fi = int(fade_in_sec  * sr)
        fo = int(fade_out_sec * sr)
        if fi > 0 and fi < len(y):
            y[:fi] *= np.linspace(0, 1, fi)
        if fo > 0 and fo < len(y):
            y[-fo:] *= np.linspace(1, 0, fo)
        return _save(y, sr, "fade")
    except Exception as exc:
        print(f"Fade Error: {exc}")
        return file_path


def beat_sync_mix(files: list) -> str | None:
    """Tempo-match and mix multiple tracks."""
    try:
        audios, target_tempo, sr = [], None, None
        for f in files:
            path = _resolve_path(f) if not isinstance(f, str) else f
            if not path or not os.path.exists(path):
                print(f"beat_sync_mix: skipping missing file: {path}")
                continue
            y, sr = librosa.load(path, sr=None)

            # librosa 0.10 returns tempo as ndarray — coerce to scalar float
            tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo_raw)[0])

            if target_tempo is None:
                target_tempo = tempo

            if tempo > 0 and target_tempo > 0:
                rate = target_tempo / tempo
            else:
                rate = 1.0

            # time_stretch requires rate > 0; skip stretch if effectively 1.0
            if abs(rate - 1.0) > 1e-3:
                # Ensure audio is long enough for time_stretch (needs > 1 frame)
                min_samples = 2048
                if len(y) >= min_samples:
                    y = librosa.effects.time_stretch(y, rate=rate)

            audios.append(y)

        if not audios:
            print("beat_sync_mix: no valid audio loaded")
            return None

        max_len = max(len(a) for a in audios)
        padded  = [np.pad(a, (0, max_len - len(a))) for a in audios]
        mixed   = np.mean(padded, axis=0)
        return _save(mixed, sr, "beatmix")
    except Exception as exc:
        print(f"Beat-sync Error: {exc}")
        return None


def get_audio_info(file_path: str) -> dict:
    """Return basic audio metadata for display."""
    try:
        path = _resolve_path(file_path) if not isinstance(file_path, str) else file_path
        y, sr = librosa.load(path, sr=None)
        duration   = librosa.get_duration(y=y, sr=sr)
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo      = float(np.atleast_1d(tempo_raw)[0])
        rms        = float(np.sqrt(np.mean(y ** 2)))
        db         = round(20 * np.log10(rms + 1e-9), 1)
        return {
            "duration": f"{duration:.1f}s",
            "bpm":      f"{tempo:.0f} BPM",
            "rms_db":   f"{db} dBFS",
            "sr":       f"{sr // 1000}kHz",
        }
    except:
        return {}

import os, uuid, time
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print("🚀 Sonic AI Music Studio — FINAL starting …")

import gradio as gr
import soundfile as sf

from generation import (
    generate_music, get_device_info,
    generate_vocals_from_lyrics,
    extract_vocals_from_audio,
    mix_vocal_tracks,
    VOICE_PRESETS,
)
from utils import (
    build_prompt, build_vocal_backing_prompt,
    rhythm_from_lyrics, melody_hint, format_status,
)
from audio_tools import (
    change_pitch, change_tempo, normalize_audio,
    apply_fade, beat_sync_mix, get_audio_info,
)

try:
    from genre_model import predict_genre
    GENRE_MODEL_OK = True
    print("✅ Genre model loaded")
except Exception as _e:
    GENRE_MODEL_OK = False
    print(f"⚠️  Genre model unavailable: {_e}")

# ── Constants ─────────────────────────────────────────────────────────────────
GENRE_OPTS = ["Classical","Blues","Rock","Jazz","Pop","Electronic",
              "Hip Hop","Lo-Fi","Ambient","Cinematic","R&B","Folk",
              "Metal","Reggae","Country"]
MOOD_OPTS  = ["Happy","Calm","Sad","Energetic","Dark","Romantic","Mysterious","Epic"]
ENERGY_OPTS = ["Low","Medium","High"]
SOUND_OPTS  = ["Pleasant to ear","Soft & smooth","Medium to ear",
               "Aggressive & loud","Harsh to ear"]
INSTRUMENT_OPTS = [
    "Piano","Violin","Acoustic Guitar","Electric Guitar","Drums","Bass Guitar",
    "Synthesizer","Saxophone","Trumpet","Flute","Cello","Harp","Organ",
    "Keyboard","Vibraphone","Marimba","Djembe","Tabla","French Horn",
    "Trombone","Oboe","Clarinet","Viola","Double Bass","Ukulele","Banjo",
    "Sitar","Accordion","Harmonica","Pan Flute","Celesta","Harpsichord",
    "Xylophone","Glockenspiel","Didgeridoo","Bagpipes",
]
VOICE_OPTS = list(VOICE_PRESETS.keys())

os.makedirs("outputs", exist_ok=True)

# ── Shared last-run state (for re-generate) ───────────────────────────────────
_last_run: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(y, sr, prefix="out"):
    path = f"outputs/{prefix}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(path, y, sr)
    return path

def _fp(f):
    """Safely extract file path from Gradio File object or plain string.

    Handles all Gradio versions:
      - Gradio 4.x  → FileData dataclass  → use .path (actual temp path)
      - Gradio 4.x  → dict format         → use ["path"] or ["name"]
      - Gradio 3.x  → NamedTemporaryFile  → use .name
      - plain str / Path                  → convert to str
    """
    if f is None:
        return None
    # Gradio 4.x: FileData has .path = actual temp file path
    if hasattr(f, "path") and f.path:
        return str(f.path)
    # Gradio 4.x dict representation (some sub-versions)
    if isinstance(f, dict):
        p = f.get("path") or f.get("name") or f.get("url") or f.get("tmp_path")
        return str(p) if p else None
    # Gradio 3.x: NamedTemporaryFile has .name = full path
    if hasattr(f, "name"):
        return str(f.name)
    # Fallback: treat as string directly
    return str(f)

def _ts():
    return time.strftime("%H:%M:%S")

def _status(step, detail="", ok=False):
    icon = "✅" if ok else "◈"
    msg  = f"[{_ts()}] {icon} {step}"
    if detail:
        msg += f"\n    {detail}"
    return msg


def plot_waveform(file_path):
    """Return waveform + spectrogram PNG path, or None."""
    try:
        y, sr    = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        t        = np.linspace(0, duration, len(y))

        fig, axes = plt.subplots(2, 1, figsize=(13, 4), facecolor="#060a14")
        fig.subplots_adjust(hspace=0.5, left=0.06, right=0.97, top=0.88, bottom=0.12)

        # Waveform
        ax1 = axes[0]
        ax1.set_facecolor("#0a1020")
        ax1.plot(t, y, color="#00f5d4", linewidth=0.5, alpha=0.9)
        ax1.fill_between(t, y, alpha=0.10, color="#00f5d4")
        ax1.axhline(0, color="#1e3a4a", linewidth=0.5, linestyle="--")
        ax1.set_xlim(0, duration); ax1.set_ylim(-1.05, 1.05)
        ax1.set_title("WAVEFORM", color="#00f5d4", fontsize=8,
                      fontweight="bold", loc="left", pad=5, fontfamily="monospace")
        ax1.tick_params(colors="#2a5060", labelsize=7)
        for sp in ax1.spines.values(): sp.set_color("#0e2030")
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:.0f}s"))

        # Spectrogram
        ax2 = axes[1]
        ax2.set_facecolor("#0a1020")
        D  = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        im = ax2.imshow(D, aspect="auto", origin="lower", cmap="magma",
                        extent=[0, duration, 0, sr/2/1000], vmin=-80, vmax=0)
        ax2.set_title("SPECTROGRAM", color="#f72585", fontsize=8,
                      fontweight="bold", loc="left", pad=5, fontfamily="monospace")
        ax2.set_ylabel("kHz", color="#2a5060", fontsize=7)
        ax2.tick_params(colors="#2a5060", labelsize=7)
        for sp in ax2.spines.values(): sp.set_color("#0e2030")
        cb = plt.colorbar(im, ax=ax2, pad=0.01, shrink=0.8, label="dB")
        cb.ax.yaxis.set_tick_params(color="#2a5060", labelsize=6)
        cb.set_label("dB", color="#2a5060", fontsize=6)

        path = f"outputs/wave_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#060a14")
        plt.close(fig)
        return path
    except Exception as exc:
        print(f"Plot error: {exc}")
        return None


# ── Streaming music helper ────────────────────────────────────────────────────
def _stream_music(prompt, duration_sec, temperature, status_prefix="Generating"):
    """Yield (path, status_msg) tuples as chunks accumulate."""
    chunk_sec = 5
    sr        = 32000
    full      = []
    chunks    = max(1, int(duration_sec / chunk_sec))

    for i in range(chunks):
        msg = _status(status_prefix, f"chunk {i+1}/{chunks} · {chunk_sec}s each …")
        yield None, msg

        part = generate_music(prompt, chunk_sec, temperature=temperature)
        if part is None:
            yield None, _status("⚠️  Chunk failed", f"chunk {i+1} skipped — continuing")
            continue

        y, _ = librosa.load(part, sr=sr)
        full.append(y)
        combined = np.concatenate(full)
        path     = _save(combined, sr, "stream")
        yield path, _status(status_prefix, f"chunk {i+1}/{chunks} done ✓")

    if not full:
        yield None, _status("❌ Generation failed", "No audio produced. Check console.")
    else:
        combined = np.concatenate(full)
        yield _save(combined, sr, "stream"), _status(status_prefix, "all chunks merged ✓", ok=True)


# ── CORE PROCESS ─────────────────────────────────────────────────────────────
def process(
    mode,
    # Tab 1 — Music
    m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
    m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
    # Tab 2 — Vocals
    m2_lyrics_text, m2_lyrics_file, m2_voice,
    m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration,
    m2_vocal_audio,                     # ← uploaded audio for vocal extraction
    m2_vocal_gain, m2_music_gain,
    # Tab 3 — Convert
    m3_audio_files,
    # Tab 4 — Studio
    m4_file, m4_pitch, m4_tempo, m4_fade,
    # Re-gen seed override
    _seed_override=None,
):
    global _last_run
    final_out = None

    # ═══════════════════════════════════════
    # MODE 1 — CREATE MUSIC
    # ═══════════════════════════════════════
    if mode == "create_music":
        if not m1_genre and not m1_text.strip():
            yield None, _status("⚠️  Input needed", "Select a Genre or enter a Prompt"), None, ""
            return

        g  = ", ".join(m1_genre)       if m1_genre       else ""
        m  = ", ".join(m1_mood)        if m1_mood        else ""
        e  = ", ".join(m1_energy)      if m1_energy      else ""
        s  = ", ".join(m1_sound)       if m1_sound       else ""
        i  = ", ".join(m1_instruments) if m1_instruments else ""
        mh = melody_hint(m1_text)

        prompt = build_prompt(g, m, m1_text, e, s, i)
        if mh: prompt += f", {mh}"

        _last_run = dict(mode="create_music",
                         m1_text=m1_text, m1_genre=m1_genre, m1_mood=m1_mood,
                         m1_energy=m1_energy, m1_sound=m1_sound, m1_instruments=m1_instruments,
                         m1_duration=m1_duration, m1_pitch=m1_pitch, m1_tempo=m1_tempo,
                         m1_temperature=m1_temperature, m1_fade=m1_fade, prompt=prompt)

        yield None, _status("Prompt built", prompt[:120]), None, ""
        for path, msg in _stream_music(prompt, m1_duration, m1_temperature):
            final_out = path or final_out
            yield final_out, msg, None, ""

    # ═══════════════════════════════════════
    # MODE 2 — VOCAL SONG
    # ═══════════════════════════════════════
    elif mode == "vocal":
        # Collect lyrics
        lyrics = ""
        if m2_lyrics_file:
            try:
                p = _fp(m2_lyrics_file)
                with open(p, "r", encoding="utf-8") as f:
                    lyrics = f.read()
            except Exception as ex:
                print(f"Lyrics file read error: {ex}")
        if m2_lyrics_text and m2_lyrics_text.strip():
            lyrics = m2_lyrics_text

        if not lyrics.strip():
            yield None, _status("⚠️  Lyrics needed", "Enter lyrics or upload a .txt file"), None, ""
            return

        g = ", ".join(m2_genre)       if m2_genre       else ""
        m = ", ".join(m2_mood)        if m2_mood        else ""
        e = ", ".join(m2_energy)      if m2_energy      else ""
        i = ", ".join(m2_instruments) if m2_instruments else ""
        rh = rhythm_from_lyrics(lyrics)
        mh = melody_hint(lyrics)

        _last_run = dict(mode="vocal",
                         m2_lyrics_text=lyrics, m2_lyrics_file=None,
                         m2_voice=m2_voice, m2_genre=m2_genre, m2_mood=m2_mood,
                         m2_energy=m2_energy, m2_instruments=m2_instruments,
                         m2_duration=m2_duration, m2_vocal_audio=m2_vocal_audio,
                         m2_vocal_gain=m2_vocal_gain, m2_music_gain=m2_music_gain)

        # ── Step 1: Instrumental backing ──────────────────────────────────────
        backing_prompt = build_vocal_backing_prompt(g, m, e, i)
        if rh: backing_prompt += f", {rh}"
        if mh: backing_prompt += f", {mh}"

        yield None, _status("Step 1/3", "Generating instrumental backing track …"), None, ""
        backing_path = None
        for path, msg in _stream_music(backing_prompt, m2_duration, 1.0, "Backing"):
            backing_path = path or backing_path
            yield backing_path, msg, None, ""

        if backing_path is None:
            yield None, _status("❌ Backing failed", "Could not generate instrumental. Check console."), None, ""
            return

        uploaded_vocal_fp = _fp(m2_vocal_audio)

        # ── Step 2a: Extract vocals from uploaded audio (if provided) ─────────
        if uploaded_vocal_fp and os.path.exists(uploaded_vocal_fp):
            yield backing_path, _status("Step 2/3", "Extracting vocals from your uploaded audio …"), None, ""
            extracted_vpath, _ = extract_vocals_from_audio(uploaded_vocal_fp)

            if extracted_vpath:
                yield backing_path, _status("Step 3/3", "Mixing extracted vocals with AI backing …"), None, ""
                mixed = mix_vocal_tracks(
                    extracted_vpath, backing_path,
                    vocal_gain=float(m2_vocal_gain),
                    music_gain=float(m2_music_gain),
                )
                final_out = mixed if mixed else backing_path
                yield final_out, _status("Vocals blended ✓",
                    "Extracted vocals from your audio mixed with new AI music"), None, ""
            else:
                yield backing_path, _status("⚠️  Extraction failed",
                    "Falling back to Bark AI vocal synthesis …"), None, ""
                uploaded_vocal_fp = None  # fall through to Bark

        # ── Step 2b: Bark AI vocal synthesis from lyrics ──────────────────────
        if not uploaded_vocal_fp or final_out is None:
            yield backing_path, _status(
                "Step 2/3", f"Generating AI singing vocals with Bark …\n"
                            f"    Voice: {m2_voice}  |  ⏱ 30-120s on CPU, please wait"), None, ""

            vocal_path = generate_vocals_from_lyrics(lyrics, voice_key=m2_voice)

            if vocal_path is None:
                yield backing_path, _status(
                    "⚠️  Vocals failed",
                    "Bark vocal synthesis failed — returning instrumental only.\n"
                    "    Check console for details."), None, ""
                final_out = backing_path
            else:
                yield backing_path, _status("Step 3/3", "Mixing Bark vocals with backing …"), None, ""
                mixed = mix_vocal_tracks(
                    vocal_path, backing_path,
                    vocal_gain=float(m2_vocal_gain),
                    music_gain=float(m2_music_gain),
                )
                final_out = mixed if mixed else backing_path

    # ═══════════════════════════════════════
    # MODE 3 — CONVERT & MIX
    # ═══════════════════════════════════════
    elif mode == "convert":
        if not m3_audio_files:
            yield None, _status("⚠️  No files", "Upload at least one audio file"), None, ""
            return

        # Extract paths and filter out any that are None or don't exist on disk
        raw_paths = [_fp(f) for f in m3_audio_files]
        files = [p for p in raw_paths if p and os.path.exists(p)]

        if not files:
            yield None, _status("⚠️  File read error",
                                 "Could not read the uploaded files — please re-upload and try again."), None, ""
            return

        yield None, _status("Processing", f"{len(files)} valid file(s) ready"), None, ""

        if len(files) == 1:
            final_out = normalize_audio(files[0])
            if final_out is None:
                yield None, _status("❌ Process failed", "Could not normalise the audio file"), None, ""
                return
        else:
            yield None, _status("Beat-sync mix", f"Tempo-matching {len(files)} tracks …"), None, ""
            final_out = beat_sync_mix(files)
            if final_out is None:
                # Fallback: simple overlay mix without beat-sync
                yield None, _status("⚠️  Beat-sync failed",
                                     "Falling back to simple overlay mix …"), None, ""
                from audio_tools import mix_audios
                final_out = mix_audios(files)
            if final_out is None:
                yield None, _status("❌ Mix failed", "Could not mix the files — check they are valid audio"), None, ""
                return

        yield final_out, _status("Mix complete", "Applying master normalize …"), None, ""

    # ═══════════════════════════════════════
    # MODE 4 — AUDIO STUDIO
    # ═══════════════════════════════════════
    elif mode == "studio":
        fp = _fp(m4_file)
        if not fp:
            yield None, _status("⚠️  No file", "Upload an audio file first"), None, ""
            return
        final_out = fp
        yield None, _status("Loaded", os.path.basename(fp)), None, ""

    # ── POST-PROCESSING ────────────────────────────────────────────────────────
    if final_out is None:
        yield None, _status("❌ Nothing generated", "No audio output produced"), None, ""
        return

    try:
        if mode == "studio":
            if m4_pitch != 0:
                yield final_out, _status("Pitch shift", f"{m4_pitch:+.1f} semitones"), None, ""
                final_out = change_pitch(final_out, m4_pitch)
            if m4_tempo != 1.0:
                yield final_out, _status("Tempo stretch", f"×{m4_tempo:.2f}"), None, ""
                final_out = change_tempo(final_out, m4_tempo)
            if m4_fade:
                final_out = apply_fade(final_out)
        else:
            pitch_val = m1_pitch if mode == "create_music" else 0
            tempo_val = m1_tempo if mode == "create_music" else 1.0
            fade_val  = m1_fade  if mode in ("create_music",) else False

            if pitch_val != 0:
                yield final_out, _status("Pitch shift", f"{pitch_val:+.1f} semitones"), None, ""
                final_out = change_pitch(final_out, float(pitch_val))
            if tempo_val != 1.0:
                yield final_out, _status("Tempo stretch", f"×{tempo_val:.2f}"), None, ""
                final_out = change_tempo(final_out, float(tempo_val))
            if fade_val:
                final_out = apply_fade(final_out)

        yield final_out, _status("Mastering", "Peak normalizing …"), None, ""
        final_out = normalize_audio(final_out)

        info     = get_audio_info(final_out)
        info_str = "  ·  ".join(f"{k.upper()}: {v}" for k, v in info.items())

        yield final_out, _status("Visualising", "Rendering waveform + spectrogram …"), None, ""
        wave = plot_waveform(final_out)

        yield final_out, _status("COMPLETE", info_str, ok=True), wave, info_str

    except Exception as exc:
        print(f"Post-processing error: {exc}")
        yield final_out, _status("⚠️  Post error", str(exc)), None, ""


# ── Mode wrappers (keeps gradio inspect happy) ────────────────────────────────
def _pack(*a): return a   # passthrough

def run_music(m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
              m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
              m2_lyrics_text, m2_lyrics_file, m2_voice,
              m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
              m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade):
    yield from process("create_music",
        m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
        m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
        m2_lyrics_text, m2_lyrics_file, m2_voice,
        m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
        m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade)

def run_vocal(m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
              m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
              m2_lyrics_text, m2_lyrics_file, m2_voice,
              m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
              m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade):
    yield from process("vocal",
        m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
        m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
        m2_lyrics_text, m2_lyrics_file, m2_voice,
        m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
        m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade)

def run_convert(m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
                m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
                m2_lyrics_text, m2_lyrics_file, m2_voice,
                m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
                m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade):
    yield from process("convert",
        m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
        m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
        m2_lyrics_text, m2_lyrics_file, m2_voice,
        m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
        m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade)

def run_studio(m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
               m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
               m2_lyrics_text, m2_lyrics_file, m2_voice,
               m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
               m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade):
    yield from process("studio",
        m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
        m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
        m2_lyrics_text, m2_lyrics_file, m2_voice,
        m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
        m2_vocal_gain, m2_music_gain, m3_audio_files, m4_file, m4_pitch, m4_tempo, m4_fade)


# ── Genre detection ───────────────────────────────────────────────────────────
def detect_genre_fn(audio_file):
    if not GENRE_MODEL_OK:
        return "⚠️  Genre model not loaded"
    if not audio_file:
        return "⚠️  Upload a file first"
    try:
        return f"◈  DETECTED GENRE: {predict_genre(_fp(audio_file)).upper()}"
    except Exception as exc:
        return f"❌ Error: {exc}"


# ── File preview helpers ──────────────────────────────────────────────────────
def handle_upload_preview(files):
    if not files:
        return gr.update(choices=[], value=None), None
    valid = [(os.path.basename(_fp(f)), _fp(f)) for f in files[:10] if _fp(f)]
    if not valid:
        return gr.update(choices=[], value=None), None
    names = [n for n, _ in valid]
    return gr.update(choices=names, value=names[0]), valid[0][1]

def preview_selected(name, files):
    if not files or not name:
        return None
    for f in files:
        p = _fp(f)
        if p and os.path.basename(p) == name:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CSS  — Industrial Neon-Noir Studio
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
  --void:        #03050c;
  --deep:        #070c18;
  --surface:     #0c1428;
  --card:        rgba(10, 18, 38, 0.88);
  --inp:         rgba(4, 10, 28, 0.92);

  --teal:        #00f5d4;
  --teal-dim:    rgba(0,245,212,0.18);
  --teal-faint:  rgba(0,245,212,0.06);
  --pink:        #f72585;
  --pink-dim:    rgba(247,37,133,0.20);
  --amber:       #ffbe0b;
  --amber-dim:   rgba(255,190,11,0.20);
  --blue:        #3a86ff;
  --blue-dim:    rgba(58,134,255,0.20);

  --glow-t:  0 0 14px rgba(0,245,212,0.6), 0 0 50px rgba(0,245,212,0.18);
  --glow-p:  0 0 14px rgba(247,37,133,0.6), 0 0 50px rgba(247,37,133,0.18);

  --border:       rgba(0,245,212,0.12);
  --border-hover: rgba(0,245,212,0.40);
  --border-pink:  rgba(247,37,133,0.30);

  --txt-hi:  #dff6f0;
  --txt-mid: #8ab5c0;
  --txt-lo:  #2e5060;

  --font-head: 'Syne', sans-serif;
  --font-body: 'DM Sans', sans-serif;
  --font-mono: 'Space Mono', monospace;
  --r: 10px;
}

::-webkit-scrollbar { width: 4px; background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--teal-dim); border-radius: 99px; }

body, .gradio-container {
  background: var(--void) !important;
  font-family: var(--font-body) !important;
  color: var(--txt-mid) !important;
}

.gradio-container {
  max-width: 1340px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(ellipse 90% 55% at 50% -8%, rgba(0,245,212,0.05) 0%, transparent 65%),
    radial-gradient(ellipse 60% 40% at 92% 85%, rgba(247,37,133,0.05) 0%, transparent 60%),
    var(--void) !important;
}

/* Grid overlay */
.gradio-container::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(0,245,212,0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,212,0.018) 1px, transparent 1px);
  background-size: 48px 48px;
  animation: gridPulse 10s ease-in-out infinite;
}
@keyframes gridPulse { 0%,100%{opacity:.5} 50%{opacity:1} }

/* ── HEADER ── */
.sonic-header {
  text-align: center;
  padding: 52px 24px 32px;
  position: relative;
}
.sonic-logo {
  font-family: var(--font-head);
  font-size: clamp(2.6rem, 6vw, 4.2rem);
  font-weight: 800;
  letter-spacing: .06em;
  background: linear-gradient(125deg, var(--teal) 0%, #00c8ff 38%, var(--pink) 72%, var(--amber) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  line-height: 1.05;
  animation: logoShine 6s ease-in-out infinite;
  filter: drop-shadow(0 0 24px rgba(0,245,212,0.28));
}
@keyframes logoShine { 0%,100%{filter:drop-shadow(0 0 18px rgba(0,245,212,0.25))} 50%{filter:drop-shadow(0 0 36px rgba(0,245,212,0.55))} }

.sonic-sub {
  font-family: var(--font-mono);
  font-size: .75rem;
  color: var(--txt-lo);
  letter-spacing: .35em;
  text-transform: uppercase;
  margin-top: 10px;
}

/* EQ bars */
.eq-bars { display:flex; align-items:flex-end; justify-content:center; gap:5px; height:38px; margin:22px auto 0; }
.eq-bar  { width:5px; border-radius:3px 3px 0 0; animation:eqBounce 1.1s ease-in-out infinite; }
@keyframes eqBounce { 0%,100%{transform:scaleY(.3)} 50%{transform:scaleY(1)} }
.eq-bar:nth-child(1){background:var(--teal);  animation-delay:0.00s; height:55%}
.eq-bar:nth-child(2){background:var(--teal);  animation-delay:0.09s; height:85%}
.eq-bar:nth-child(3){background:#00c8ff;      animation-delay:0.18s; height:40%}
.eq-bar:nth-child(4){background:#00c8ff;      animation-delay:0.13s; height:100%}
.eq-bar:nth-child(5){background:var(--pink);  animation-delay:0.22s; height:70%}
.eq-bar:nth-child(6){background:var(--pink);  animation-delay:0.30s; height:50%}
.eq-bar:nth-child(7){background:var(--amber); animation-delay:0.07s; height:90%}
.eq-bar:nth-child(8){background:var(--amber); animation-delay:0.25s; height:62%}
.eq-bar:nth-child(9){background:var(--teal);  animation-delay:0.15s; height:75%}
.eq-bar:nth-child(10){background:#00c8ff;     animation-delay:0.04s; height:45%}
.eq-bar:nth-child(11){background:var(--pink); animation-delay:0.33s; height:88%}
.eq-bar:nth-child(12){background:var(--teal); animation-delay:0.20s; height:60%}

/* ── STATS BAR ── */
.stats-bar {
  display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;
  padding: 0 24px 32px;
}
.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 10px 20px;
  text-align: center;
  min-width: 100px;
  transition: border-color .3s, box-shadow .3s;
}
.stat-card:hover { border-color: var(--border-hover); box-shadow: var(--glow-t); }
.stat-val { font-family: var(--font-mono); font-size: .8rem; color: var(--teal); font-weight: 700; }
.stat-lbl { font-size: .65rem; color: var(--txt-lo); text-transform: uppercase; letter-spacing: .1em; margin-top: 3px; }

/* ── TABS ── */
.tab-nav { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
.tab-nav button {
  font-family: var(--font-mono) !important;
  font-size: .75rem !important;
  letter-spacing: .08em !important;
  color: var(--txt-lo) !important;
  padding: 10px 20px !important;
  border: none !important;
  background: transparent !important;
  transition: color .25s !important;
}
.tab-nav button:hover { color: var(--teal) !important; }
.tab-nav button.selected {
  color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
  text-shadow: var(--glow-t);
}

/* ── CARDS / PANELS ── */
.panel-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 20px 22px;
  margin-bottom: 14px;
}
.panel-title {
  font-family: var(--font-head);
  font-size: .9rem;
  font-weight: 700;
  color: var(--teal);
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--teal-faint);
}

/* ── GRADIO COMPONENTS ── */
label, .gradio-label { font-family: var(--font-mono) !important; font-size:.72rem !important; color:var(--txt-lo) !important; letter-spacing:.08em !important; text-transform:uppercase !important; }

input, textarea, .gradio-textbox textarea {
  background: var(--inp) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--txt-hi) !important;
  font-family: var(--font-body) !important;
  font-size: .9rem !important;
  transition: border-color .25s, box-shadow .25s !important;
}
input:focus, textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px var(--teal-dim) !important;
  outline: none !important;
}

.gradio-dropdown > div {
  background: var(--inp) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--txt-hi) !important;
}
.gradio-slider input[type=range] { accent-color: var(--teal) !important; }

/* ── BUTTONS ── */
.btn-generate {
  background: linear-gradient(120deg, #005c52 0%, #007a6e 50%, #00a08e 100%) !important;
  border: 1px solid var(--teal) !important;
  border-radius: var(--r) !important;
  color: #fff !important;
  font-family: var(--font-head) !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  letter-spacing: .15em !important;
  text-transform: uppercase !important;
  padding: 16px 40px !important;
  cursor: pointer !important;
  box-shadow: var(--glow-t) !important;
  transition: all .3s !important;
  width: 100% !important;
}
.btn-generate:hover {
  background: linear-gradient(120deg, #00756a 0%, #009d8e 50%, #00c8b0 100%) !important;
  box-shadow: 0 0 24px rgba(0,245,212,0.9), 0 0 80px rgba(0,245,212,0.35) !important;
  transform: translateY(-1px) !important;
}
.btn-regen {
  background: linear-gradient(120deg, #1a0a30 0%, #28105a 50%, #3a1878 100%) !important;
  border: 1px solid var(--pink) !important;
  border-radius: var(--r) !important;
  color: #fff !important;
  font-family: var(--font-head) !important;
  font-size: .88rem !important;
  font-weight: 700 !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  padding: 13px 28px !important;
  cursor: pointer !important;
  box-shadow: var(--glow-p) !important;
  transition: all .3s !important;
  width: 100% !important;
}
.btn-regen:hover {
  background: linear-gradient(120deg, #2a1248 0%, #3c1a78 50%, #5224a0 100%) !important;
  box-shadow: 0 0 22px rgba(247,37,133,0.9), 0 0 70px rgba(247,37,133,0.35) !important;
  transform: translateY(-1px) !important;
}
.btn-secondary {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--txt-mid) !important;
  font-family: var(--font-mono) !important;
  font-size: .72rem !important;
  padding: 8px 16px !important;
  cursor: pointer !important;
  transition: all .25s !important;
}
.btn-secondary:hover { border-color: var(--border-hover) !important; color: var(--teal) !important; }

/* ── OUTPUT PANEL ── */
.output-panel {
  background: var(--card);
  border: 1px solid var(--border-pink);
  border-radius: 14px;
  padding: 24px;
  margin-top: 20px;
  box-shadow: 0 0 40px rgba(247,37,133,0.06);
}
.output-title {
  font-family: var(--font-head);
  font-size: 1rem;
  font-weight: 700;
  color: var(--pink);
  letter-spacing: .15em;
  text-transform: uppercase;
  margin-bottom: 18px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-pink);
}

/* Status box */
#status-box textarea {
  font-family: var(--font-mono) !important;
  font-size: .72rem !important;
  color: var(--teal) !important;
  background: #030710 !important;
  border-color: var(--teal-dim) !important;
  line-height: 1.7 !important;
}

/* Info badge */
.info-badge {
  background: rgba(0,245,212,0.07);
  border: 1px solid var(--teal-dim);
  border-radius: 8px;
  padding: 12px 16px;
  font-family: var(--font-mono);
  font-size: .72rem;
  color: var(--txt-mid);
  line-height: 1.7;
}
.info-badge b { color: var(--teal); }

/* Vocal how-it-works box */
.how-it-works {
  background: linear-gradient(120deg, rgba(247,37,133,0.06), rgba(58,134,255,0.06));
  border: 1px solid rgba(247,37,133,0.18);
  border-radius: 10px;
  padding: 16px 20px;
  font-size: .84rem;
  line-height: 1.75;
  color: var(--txt-mid);
  margin-bottom: 14px;
}
.how-it-works b { color: var(--pink); }
.how-it-works .step { color: var(--teal); font-family: var(--font-mono); }

/* Accordion */
.gradio-accordion > div:first-child {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--txt-mid) !important;
  font-family: var(--font-mono) !important;
  font-size: .75rem !important;
}

/* ── FOOTER ── */
.sonic-footer {
  text-align: center;
  padding: 28px 24px 40px;
  font-family: var(--font-mono);
  font-size: .68rem;
  color: var(--txt-lo);
  letter-spacing: .12em;
  border-top: 1px solid var(--border);
  margin-top: 32px;
}
.sonic-footer .hl { color: var(--teal); }

/* Markdown headings inside gradio */
.gradio-markdown h2 {
  font-family: var(--font-head) !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  color: var(--teal) !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  margin: 0 0 14px !important;
  padding-bottom: 8px !important;
  border-bottom: 1px solid var(--teal-faint) !important;
}
"""

device_info = get_device_info()

# ─────────────────────────────────────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Sonic AI · Music Studio") as app:

    # ── HEADER ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="sonic-header">
      <div class="sonic-logo">SONIC&nbsp;AI</div>
      <div class="sonic-sub">⟡ Industrial Music Generation Studio ⟡</div>
      <div class="eq-bars">
        {''.join('<div class="eq-bar"></div>' * 12)}
      </div>
    </div>
    <div class="stats-bar">
      <div class="stat-card"><div class="stat-val">MusicGen</div><div class="stat-lbl">Music AI</div></div>
      <div class="stat-card"><div class="stat-val">Bark</div><div class="stat-lbl">Vocal AI</div></div>
      <div class="stat-card"><div class="stat-val">HPSS</div><div class="stat-lbl">Vocal Extract</div></div>
      <div class="stat-card"><div class="stat-val">{device_info["device"].split("·")[0].strip()}</div><div class="stat-lbl">Device</div></div>
      <div class="stat-card"><div class="stat-val">32 kHz</div><div class="stat-lbl">Sample Rate</div></div>
      <div class="stat-card"><div class="stat-val">WAV</div><div class="stat-lbl">Format</div></div>
    </div>
    """)

    # ── TABS ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ══════════════════════════════════════════════
        # TAB 1 — CREATE MUSIC
        # ══════════════════════════════════════════════
        with gr.Tab("🎵  Create Music"):
            gr.Markdown("## ◈ Music Generation")
            with gr.Row():
                with gr.Column(scale=3):
                    m1_text = gr.Textbox(
                        label="Creative Prompt",
                        placeholder="e.g.  'A melancholic piano piece with soft strings on a rainy evening'",
                        lines=3)
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="info-badge">
                      <b>Prompt Tips</b><br>
                      · Describe mood, scene, atmosphere<br>
                      · Mention tempo or dynamics<br>
                      · Combine with selectors below
                    </div>""")

            with gr.Row():
                m1_genre  = gr.Dropdown(GENRE_OPTS,  multiselect=True, label="Genre",       scale=2)
                m1_mood   = gr.Dropdown(MOOD_OPTS,   multiselect=True, label="Mood",         scale=2)
                m1_energy = gr.Dropdown(ENERGY_OPTS, multiselect=True, label="Energy Level", scale=1)

            with gr.Row():
                m1_sound       = gr.Dropdown(SOUND_OPTS,      multiselect=True, label="Sound Character")
                m1_instruments = gr.Dropdown(INSTRUMENT_OPTS, multiselect=True, label="Instruments")

            with gr.Accordion("⚙  Advanced Controls", open=False):
                with gr.Row():
                    m1_duration    = gr.Slider(10, 300, value=60,  step=5,   label="Duration (seconds)")
                    m1_temperature = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Creativity / Temperature")
                with gr.Row():
                    m1_pitch = gr.Slider(-12, 12, value=0,   step=0.5, label="Pitch Shift (semitones)")
                    m1_tempo = gr.Slider(0.5, 2.5, value=1.0, step=0.05, label="Tempo Multiplier")
                m1_fade = gr.Checkbox(label="Apply Fade-in / Fade-out", value=True)

            with gr.Row():
                btn_music = gr.Button("⟡  GENERATE MUSIC  ⟡", elem_classes="btn-generate", size="lg")
                btn_regen_music = gr.Button("↺  RE-GENERATE", elem_classes="btn-regen", size="lg")

        # ══════════════════════════════════════════════
        # TAB 2 — VOCAL SONG  (fully fixed)
        # ══════════════════════════════════════════════
        with gr.Tab("🎤  Vocal Song"):
            gr.Markdown("## ◈ Vocal Song Creation")
            gr.HTML("""
            <div class="how-it-works">
              <b>How vocal generation works — 3 automatic steps:</b><br>
              <span class="step">STEP 1</span> · MusicGen creates a matching <b>instrumental backing track</b><br>
              <span class="step">STEP 2</span> · <b>Option A</b> — Upload your own audio → HPSS extracts your vocal layer and blends it into the new music<br>
              <span class="step">      </span> · <b>Option B</b> — No upload → Bark AI synthesises <b>real singing vocals</b> from your lyrics<br>
              <span class="step">STEP 3</span> · Vocals + backing are <b>mixed</b> and mastered into a final track<br>
              <br>
              ⏱ Bark vocal synthesis takes <b>30–120 s on CPU</b> — please be patient!
            </div>""")

            with gr.Row():
                with gr.Column(scale=2):
                    m2_lyrics_text = gr.Textbox(
                        label="Lyrics  (or leave blank if uploading an audio file with vocals)",
                        placeholder="Verse 1:\nSunrise breaks the darkened sky…\n\nChorus:\nFly above the clouds tonight…",
                        lines=9)
                with gr.Column(scale=1):
                    m2_lyrics_file = gr.File(file_types=[".txt"], label="Upload Lyrics (.txt)")
                    m2_voice       = gr.Dropdown(choices=VOICE_OPTS, value=VOICE_OPTS[0],
                                                  label="🎙 Bark Voice Style")
                    gr.HTML("""<div class="info-badge" style="margin-top:8px">
                      Used when <b>no audio upload</b> is provided.<br>
                      Lyrics auto-analysed for rhythm & key.
                    </div>""")

            gr.HTML('<div class="panel-card"><div class="panel-title">📂 Vocal Audio Upload (optional)</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    m2_vocal_audio = gr.File(
                        file_types=["audio"],
                        label="Upload Audio File Containing Vocals  (WAV / MP3 / OGG)")
                with gr.Column(scale=1):
                    gr.HTML("""<div class="info-badge">
                      <b>HPSS Vocal Extraction</b><br>
                      Separates the vocal/harmonic layer<br>
                      from your audio and re-mixes it<br>
                      with the AI backing track.<br><br>
                      Leave blank to use Bark AI vocals.
                    </div>""")
            gr.HTML('</div>')

            with gr.Row():
                m2_vocal_gain = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Vocal Volume")
                m2_music_gain = gr.Slider(0.1, 1.5, value=0.5, step=0.05, label="Backing Volume")

            with gr.Row():
                m2_genre      = gr.Dropdown(GENRE_OPTS,      multiselect=True, label="Genre")
                m2_mood       = gr.Dropdown(MOOD_OPTS,       multiselect=True, label="Mood")
                m2_energy     = gr.Dropdown(ENERGY_OPTS,     multiselect=True, label="Energy")
                m2_instruments = gr.Dropdown(INSTRUMENT_OPTS, multiselect=True, label="Instruments")

            m2_duration = gr.Slider(10, 120, value=30, step=5, label="Backing Duration (seconds)")

            with gr.Row():
                btn_vocal = gr.Button("⟡  GENERATE VOCAL SONG  ⟡", elem_classes="btn-generate", size="lg")
                btn_regen_vocal = gr.Button("↺  RE-GENERATE", elem_classes="btn-regen", size="lg")

        # ══════════════════════════════════════════════
        # TAB 3 — CONVERT & MIX
        # ══════════════════════════════════════════════
        with gr.Tab("🔄  Convert & Mix"):
            gr.Markdown("## ◈ Audio Conversion & Beat-Sync Mix")
            with gr.Row():
                with gr.Column(scale=2):
                    m3_audio_files = gr.File(
                        file_count="multiple", file_types=["audio"],
                        label="Upload Audio Files (up to 10)")
                with gr.Column(scale=1):
                    m3_dropdown = gr.Dropdown(label="Preview a file")
                    m3_preview  = gr.Audio(label="Preview", interactive=False)
                    m3_detect_in  = gr.File(file_types=["audio"], label="Detect Genre")
                    m3_detect_out = gr.Textbox(label="Detected Genre",
                                               interactive=False, elem_id="status-box")
                    btn_detect = gr.Button("◉ Detect Genre", elem_classes="btn-secondary", size="sm")

            gr.HTML("""<div class="info-badge" style="margin:10px 0">
              <b>Single file</b> → normalise &amp; master &nbsp;|&nbsp;
              <b>Multiple files</b> → tempo-sync all and create a beat-matched overlay mix
            </div>""")

            with gr.Row():
                btn_convert = gr.Button("⟡  PROCESS & MIX  ⟡", elem_classes="btn-generate", size="lg")

        # ══════════════════════════════════════════════
        # TAB 4 — AUDIO STUDIO
        # ══════════════════════════════════════════════
        with gr.Tab("🎛️  Audio Studio"):
            gr.Markdown("## ◈ Post-Production Studio")
            with gr.Row():
                with gr.Column(scale=2):
                    m4_file = gr.File(file_types=["audio"], label="Upload Audio File")
                with gr.Column(scale=2):
                    gr.HTML("""<div class="info-badge">
                      <b>Pitch Shift</b> — Transpose in semitones<br>
                      <b>Tempo Stretch</b> — Speed up or slow down<br>
                      <b>Fade</b> — Smooth fade-in / fade-out<br>
                      <b>Master</b> — Peak normalize to −1 dBFS
                    </div>""")
            with gr.Row():
                m4_pitch = gr.Slider(-12, 12, value=0,   step=0.5, label="Pitch Shift (semitones)")
                m4_tempo = gr.Slider(0.5, 2.5, value=1.0, step=0.05, label="Tempo Multiplier")
                m4_fade  = gr.Checkbox(label="Fade In / Out", value=True)

            with gr.Row():
                btn_studio = gr.Button("⟡  PROCESS AUDIO  ⟡", elem_classes="btn-generate", size="lg")

    # ── OUTPUT SECTION ─────────────────────────────────────────────────────────
    gr.HTML('<div class="output-panel">')
    gr.HTML('<div class="output-title">◈ Output</div>')
    with gr.Row():
        with gr.Column(scale=3):
            out_audio = gr.Audio(label="Generated Audio", interactive=False)
            wave_img  = gr.Image(label="Waveform & Spectrogram", interactive=False)
        with gr.Column(scale=1):
            status   = gr.Textbox(label="Progress / Status", lines=9,
                                  interactive=False, elem_id="status-box")
            info_out = gr.Textbox(label="Audio Info", lines=3, interactive=False)
    gr.HTML('</div>')

    # FOOTER
    gr.HTML("""
    <div class="sonic-footer">
      <span class="hl">SONIC AI</span> · Industrial Music Generation Studio ·
      Masters Major Project · MusicGen + Bark + HPSS · Gradio ·
      <span class="hl">facebook/musicgen-small</span> + <span class="hl">suno/bark-small</span>
    </div>""")

    # ── ALL INPUTS (shared by every button) ───────────────────────────────────
    ALL_INPUTS = [
        # Tab 1
        m1_text, m1_genre, m1_mood, m1_energy, m1_sound, m1_instruments,
        m1_duration, m1_pitch, m1_tempo, m1_temperature, m1_fade,
        # Tab 2
        m2_lyrics_text, m2_lyrics_file, m2_voice,
        m2_genre, m2_mood, m2_energy, m2_instruments, m2_duration, m2_vocal_audio,
        m2_vocal_gain, m2_music_gain,
        # Tab 3
        m3_audio_files,
        # Tab 4
        m4_file, m4_pitch, m4_tempo, m4_fade,
    ]
    ALL_OUTPUTS = [out_audio, status, wave_img, info_out]

    # ── Re-generate wrappers ──────────────────────────────────────────────────
    def regen_music(*a):
        """Re-run music generation with same settings (fresh random seed)."""
        yield from run_music(*a)

    def regen_vocal(*a):
        """Re-run vocal generation with same settings (fresh random seed)."""
        yield from run_vocal(*a)

    # ── Event bindings ────────────────────────────────────────────────────────
    btn_music.click(run_music,    inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)
    btn_regen_music.click(regen_music, inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)
    btn_vocal.click(run_vocal,    inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)
    btn_regen_vocal.click(regen_vocal, inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)
    btn_convert.click(run_convert, inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)
    btn_studio.click(run_studio,  inputs=ALL_INPUTS, outputs=ALL_OUTPUTS)

    m3_audio_files.upload(handle_upload_preview,
                          inputs=[m3_audio_files],
                          outputs=[m3_dropdown, m3_preview])
    m3_dropdown.change(preview_selected,
                       inputs=[m3_dropdown, m3_audio_files],
                       outputs=[m3_preview])
    btn_detect.click(detect_genre_fn,
                     inputs=[m3_detect_in],
                     outputs=[m3_detect_out])


# ── LAUNCH ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False,
        show_error=True,
        inbrowser=True,
    )

"""
utils.py — Enhanced Prompt Engineering Engine
Sonic AI Music Generator | Masters Major Project — FINAL
"""

# Mood → musical theory mappings
MOOD_MAP = {
    "Happy":      "upbeat, major key, bright chord progressions, lively rhythm",
    "Sad":        "slow tempo, minor key, melancholic progression, soft dynamics",
    "Calm":       "ambient, gentle dynamics, smooth legato, peaceful atmosphere",
    "Energetic":  "driving rhythm, high energy, strong percussive accents, dynamic range",
    "Dark":       "minor key, low register, tension-building, dissonant undertones",
    "Romantic":   "soft dynamics, lyrical phrasing, warm timbre, flowing melody",
    "Mysterious": "modal harmony, sparse texture, atmospheric reverb, ethereal tones",
    "Epic":       "orchestral swell, powerful brass, cinematic dynamics, grand scale",
}

ENERGY_MAP = {
    "Low":    "soft, minimal, ambient texture, gentle attack",
    "Medium": "moderate tempo, balanced dynamics, steady groove",
    "High":   "fast tempo, strong transients, driving percussion, full frequency",
}

SOUND_MAP = {
    "Pleasant to ear":   "warm harmonics, smooth timbre, balanced EQ",
    "Harsh to ear":      "distorted edges, raw texture, aggressive overtones",
    "Medium to ear":     "natural dynamics, moderate processing",
    "Soft & smooth":     "rounded attack, gentle compression, velvety texture",
    "Aggressive & loud": "heavy saturation, loud master, punchy transients, maximal energy",
}


def build_prompt(genre: str, mood: str, text: str, energy: str,
                 sound_feel: str, instruments: str) -> str:
    """Build a rich, structured MusicGen prompt from all parameters."""
    parts = []

    if genre:
        parts.append(f"{genre} music")
    else:
        parts.append("instrumental music")

    if mood:
        for m in [x.strip() for x in mood.split(",")]:
            if m in MOOD_MAP:
                parts.append(MOOD_MAP[m])
            elif m:
                parts.append(m.lower())

    if energy:
        for e in [x.strip() for x in energy.split(",")]:
            if e in ENERGY_MAP:
                parts.append(ENERGY_MAP[e])

    if sound_feel:
        for s in [x.strip() for x in sound_feel.split(",")]:
            if s in SOUND_MAP:
                parts.append(SOUND_MAP[s])

    if instruments:
        parts.append(f"featuring {instruments}")

    if text and text.strip():
        parts.append(f"theme: {text.strip()}")

    return ", ".join(p for p in parts if p)


def build_vocal_backing_prompt(genre: str, mood: str, energy: str, instruments: str) -> str:
    """Build a prompt for the instrumental backing behind vocals — no vocal hints."""
    base = build_prompt(genre, mood, "", energy, "", instruments)
    base += ", instrumental background music, melodic backing, no vocals, no speech"
    return base


def rhythm_from_lyrics(lyrics: str) -> str:
    """Infer rhythm hints from lyric density."""
    words = lyrics.split()
    count = len(words)
    if count < 20:
        return "slow tempo, sparse arrangement, intimate feel"
    elif count < 60:
        return "medium tempo, balanced arrangement, verse-chorus structure"
    else:
        return "upbeat tempo, dense arrangement, energetic delivery"


def melody_hint(text: str) -> str:
    """Derive melodic direction from text keywords."""
    if not text:
        return ""
    t = text.lower()
    hints = []
    if any(w in t for w in ["sad", "cry", "tear", "loss", "grief"]):
        hints.append("minor key, emotional melody")
    if any(w in t for w in ["happy", "joy", "love", "bright", "fun"]):
        hints.append("major key, bright melody")
    if any(w in t for w in ["dark", "shadow", "night", "deep"]):
        hints.append("deep bass, minor progression")
    if any(w in t for w in ["romantic", "gentle", "soft", "tender"]):
        hints.append("soft piano, smooth melody")
    if any(w in t for w in ["epic", "grand", "powerful", "battle"]):
        hints.append("orchestral build, cinematic swell")
    if any(w in t for w in ["peaceful", "calm", "ambient", "dream"]):
        hints.append("ambient pads, gentle arpeggios")
    return ", ".join(hints)


def format_status(step: str, detail: str = "", icon: str = "◈") -> str:
    """Format a status message for the UI."""
    base = f"{icon}  {step.upper()}"
    if detail:
        base += f"\n    {detail}"
    return base

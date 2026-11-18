"""
Passo 2: trascrizione con Whisper (solo testo + timestamp, ancora SENZA sapere chi parla).

Cosa fa:
  1. Prende l'mp3 scaricato al passo 1
  2. Ne ritaglia i primi N minuti e li converte in WAV mono 16 kHz
     (è il formato che sia Whisper sia pyannote vogliono come ingresso)
  3. Trascrive con faster-whisper
  4. Salva i segmenti in data/trascrizioni/<id>_whisper.json

Uso:  .venv/bin/python 02_trascrivi.py
"""

import json
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

# ── Impostazioni (le cambi qui) ───────────────────────────────────────────
ID_EPISODIO = "63134639027"   # nome del file mp3 scaricato al passo 1
MINUTI = 10                   # quanti minuti trascrivere (per il test)
MODELLO_WHISPER = "small"     # tiny / base / small / medium / large-v3 (più grande = più preciso ma più lento)

# ── Percorsi ──────────────────────────────────────────────────────────────
file_mp3 = Path(f"data/audio/{ID_EPISODIO}.mp3")
file_wav = Path(f"data/audio/{ID_EPISODIO}_primi{MINUTI}min.wav")
cartella_out = Path("data/trascrizioni")
file_json = cartella_out / f"{ID_EPISODIO}_whisper.json"
cartella_out.mkdir(parents=True, exist_ok=True)


# ── 1. Ritaglia e converti con ffmpeg ─────────────────────────────────────
# -t   = durata da tenere (in secondi)
# -ac 1 = un solo canale (mono)
# -ar 16000 = 16.000 campioni al secondo
if not file_wav.exists():
    print(f"Ritaglio i primi {MINUTI} minuti...")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(file_mp3),
         "-t", str(MINUTI * 60),
         "-ac", "1", "-ar", "16000",
         str(file_wav)],
        check=True,
    )


# ── 2. Carica il modello Whisper ──────────────────────────────────────────
# device="cpu": su Mac faster-whisper non usa la GPU, ma va bene.
# compute_type="int8": numeri più "piccoli" -> più veloce e meno memoria.
print(f"Carico il modello Whisper '{MODELLO_WHISPER}'...")
modello = WhisperModel(MODELLO_WHISPER, device="cpu", compute_type="int8")


# ── 3. Trascrivi ──────────────────────────────────────────────────────────
# language="it": evitiamo che Whisper debba indovinare la lingua.
# vad_filter=True: salta i tratti di silenzio/musica (meno "allucinazioni").
# transcribe() restituisce un generatore: la trascrizione avviene mentre lo scorriamo.
print("Trascrivo (può volerci qualche minuto)...")
segmenti_iter, info = modello.transcribe(str(file_wav), language="it", vad_filter=True)

segmenti = []
for s in segmenti_iter:
    segmenti.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
    print(f"[{s.start:7.1f} -> {s.end:7.1f}] {s.text.strip()}")


# ── 4. Salva ──────────────────────────────────────────────────────────────
with open(file_json, "w", encoding="utf-8") as f:
    json.dump(segmenti, f, ensure_ascii=False, indent=2)

print(f"\nFatto: {len(segmenti)} segmenti salvati in {file_json}")

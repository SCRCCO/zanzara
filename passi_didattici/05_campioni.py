"""
Passo 5: ritaglia dei campioni audio per ogni voce, così puoi ascoltarli e dire chi è chi.

Cosa fa:
  1. Legge i turni di parola di pyannote (passo 3)
  2. Per ogni SPEAKER_xx sceglie i turni più lunghi (sono i più "puliti")
  3. Ritaglia l'audio con ffmpeg in data/campioni/SPEAKER_xx_1.wav, _2.wav
  4. Stampa cosa viene detto in quel punto (dal passo 4) come indizio

Uso:  .venv/bin/python 05_campioni.py
"""

import json
import subprocess
from pathlib import Path

ID_EPISODIO = "63134639027"
MINUTI = 10
CAMPIONI_PER_VOCE = 2   # quanti clip ritagliare per ogni voce
DURATA_MAX = 25         # secondi: un clip più lungo non serve, basta per riconoscere una voce

file_wav = Path(f"data/audio/{ID_EPISODIO}_primi{MINUTI}min.wav")
cartella = Path("data/trascrizioni")
cartella_out = Path("data/campioni")
cartella_out.mkdir(parents=True, exist_ok=True)

turni = json.load(open(cartella / f"{ID_EPISODIO}_speaker.json", encoding="utf-8"))
frasi = json.load(open(cartella / f"{ID_EPISODIO}_unito.json", encoding="utf-8"))

for voce in sorted({t["speaker"] for t in turni}):
    # tutti i turni di questa voce, dal più lungo al più corto
    suoi = [t for t in turni if t["speaker"] == voce]
    suoi.sort(key=lambda t: t["end"] - t["start"], reverse=True)

    print(f"\n── {voce}")
    for numero, turno in enumerate(suoi[:CAMPIONI_PER_VOCE], start=1):
        inizio = turno["start"]
        durata = min(turno["end"] - turno["start"], DURATA_MAX)
        nome_file = cartella_out / f"{voce}_{numero}.wav"

        # -ss = da che secondo partire, -t = per quanti secondi
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(inizio), "-t", str(durata),
             "-i", str(file_wav), str(nome_file)],
            check=True,
        )

        # le frasi di Whisper che cadono dentro questo clip (indizio testuale)
        testo = " ".join(f["text"] for f in frasi if inizio <= f["start"] < inizio + durata)
        print(f"   {nome_file}  ({durata:.0f}s, da {inizio:.0f}s)")
        print(f"      «{testo[:160]}»")

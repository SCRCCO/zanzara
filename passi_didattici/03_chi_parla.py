"""
Passo 3: diarizzazione con pyannote = capire CHI parla e QUANDO (senza sapere cosa dice).

Cosa fa:
  1. Carica il WAV dei primi N minuti (creato al passo 2)
  2. Lo passa a pyannote, che lo divide in "turni di parola" e dà a ogni voce
     un'etichetta anonima: SPEAKER_00, SPEAKER_01, ...
  3. Salva i turni in data/trascrizioni/<id>_speaker.json
  4. Stampa quanto parla ciascuna voce (per capire a occhio chi è chi)

NB: pyannote NON sa i nomi. Non sa che SPEAKER_01 è Cruciani: lo capiremo noi
dopo, guardando chi parla di più / ascoltando un campione.

Uso:  .venv/bin/python 03_chi_parla.py
"""

import json
import os
import wave
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# ── Impostazioni ──────────────────────────────────────────────────────────
ID_EPISODIO = "63134639027"
MINUTI = 10  # deve essere uguale al passo 2, perché usiamo lo stesso WAV

file_wav = Path(f"data/audio/{ID_EPISODIO}_primi{MINUTI}min.wav")
file_json = Path(f"data/trascrizioni/{ID_EPISODIO}_speaker.json")

load_dotenv()
TOKEN_HF = os.environ["HF_KEY"]  # la chiave Hugging Face nel file .env


# ── 1. Carica l'audio in memoria ──────────────────────────────────────────
# Un WAV è una lista di numeri (16 bit) = l'ampiezza del suono ad ogni istante.
# Li leggiamo con il modulo standard "wave" e li trasformiamo in un array
# di numeri decimali tra -1 e 1, nel formato che pyannote si aspetta.
with wave.open(str(file_wav), "rb") as w:
    frequenza = w.getframerate()                       # 16000 campioni al secondo
    byte_grezzi = w.readframes(w.getnframes())

campioni = np.frombuffer(byte_grezzi, dtype=np.int16).astype(np.float32) / 32768.0
# pyannote vuole un tensore con forma (canali, campioni): noi abbiamo 1 canale
forma_onda = torch.from_numpy(campioni).unsqueeze(0)
audio = {"waveform": forma_onda, "sample_rate": frequenza}
print(f"Audio caricato: {len(campioni) / frequenza / 60:.1f} minuti")


# ── 2. Carica il modello di diarizzazione ─────────────────────────────────
# La prima volta scarica i pesi da Hugging Face (serve aver accettato le
# condizioni d'uso del modello sul sito, altrimenti errore 403).
print("Carico pyannote...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=TOKEN_HF)
pipeline.to(torch.device("cpu"))


# ── 3. Esegui la diarizzazione ────────────────────────────────────────────
print("Analizzo le voci (può volerci qualche minuto)...")
risultato = pipeline(audio)

# Nelle versioni nuove di pyannote il risultato è un oggetto che contiene
# l'annotazione dentro ".speaker_diarization"; nelle vecchie è già l'annotazione.
annotazione = getattr(risultato, "speaker_diarization", risultato)


# ── 4. Trasforma in una lista semplice e salva ────────────────────────────
# itertracks(yield_label=True) restituisce, per ogni turno: (intervallo, traccia, voce)
turni = []
for intervallo, _, voce in annotazione.itertracks(yield_label=True):
    turni.append({"start": round(intervallo.start, 2), "end": round(intervallo.end, 2), "speaker": voce})

with open(file_json, "w", encoding="utf-8") as f:
    json.dump(turni, f, ensure_ascii=False, indent=2)


# ── 5. Riepilogo: quanti secondi parla ogni voce ──────────────────────────
secondi_per_voce = {}
for t in turni:
    secondi_per_voce[t["speaker"]] = secondi_per_voce.get(t["speaker"], 0) + (t["end"] - t["start"])

print(f"\n{len(turni)} turni di parola salvati in {file_json}\n")
for voce, sec in sorted(secondi_per_voce.items(), key=lambda x: -x[1]):
    print(f"  {voce}: {sec:6.0f} s ({sec / 60:.1f} min)")

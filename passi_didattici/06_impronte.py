"""
Passo 6: calcola le "impronte vocali" (voice embeddings) di Cruciani, Parenzo e Clizia.

Idea: un modello trasforma un pezzo di voce in un vettore di numeri (l'impronta).
Voci della stessa persona -> vettori simili. Voci diverse -> vettori lontani.
Per ogni persona facciamo la MEDIA dei vettori di tutti i suoi turni, e salviamo
quella media: è la sua impronta "di riferimento".

Output: data/impronte/<nome>.npy  (un file per persona)

Uso:  .venv/bin/python 06_impronte.py
"""

import json
import os
import wave
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Inference, Model
from pyannote.core import Segment

# ── Impostazioni ──────────────────────────────────────────────────────────
ID_EPISODIO = "63134639027"
MINUTI = 10

# Chi è chi: l'abbiamo deciso ascoltando i campioni del passo 5.
# NB: queste etichette valgono SOLO per questo episodio.
NOMI = {
    "SPEAKER_02": "cruciani",
    "SPEAKER_03": "parenzo",
    "SPEAKER_05": "clizia",
}
DURATA_MIN_TURNO = 3.0  # secondi: turni più corti danno impronte inaffidabili

file_wav = Path(f"data/audio/{ID_EPISODIO}_primi{MINUTI}min.wav")
file_turni = Path(f"data/trascrizioni/{ID_EPISODIO}_speaker.json")
cartella_out = Path("data/impronte")
cartella_out.mkdir(parents=True, exist_ok=True)

load_dotenv()
TOKEN_HF = os.environ["HF_KEY"]


# ── 1. Carica l'audio (come al passo 3) ───────────────────────────────────
with wave.open(str(file_wav), "rb") as w:
    frequenza = w.getframerate()
    byte_grezzi = w.readframes(w.getnframes())
campioni = np.frombuffer(byte_grezzi, dtype=np.int16).astype(np.float32) / 32768.0
audio = {"waveform": torch.from_numpy(campioni).unsqueeze(0), "sample_rate": frequenza}


# ── 2. Carica il modello che trasforma una voce in un vettore ─────────────
# "wespeaker" è un modello addestrato apposta per riconoscere i parlanti.
# window="whole" = un solo vettore per tutto il pezzo che gli diamo
# (invece di uno ogni pochi secondi).
print("Carico il modello di impronte vocali...")
modello = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=TOKEN_HF)
estrattore = Inference(modello, window="whole")


def impronta(inizio, fine):
    """Vettore (array di numeri) che descrive la voce tra due istanti."""
    return estrattore.crop(audio, Segment(inizio, fine))


def somiglianza(a, b):
    """Somiglianza del coseno: 1 = identici, 0 = niente in comune, <0 = opposti."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── 3. Calcola l'impronta di ogni persona ─────────────────────────────────
turni = json.load(open(file_turni, encoding="utf-8"))
impronte_medie = {}

for etichetta, nome in NOMI.items():
    suoi = [t for t in turni if t["speaker"] == etichetta and t["end"] - t["start"] >= DURATA_MIN_TURNO]
    secondi = sum(t["end"] - t["start"] for t in suoi)
    print(f"\n{nome}: uso {len(suoi)} turni (totale {secondi:.0f} s)")

    vettori = [impronta(t["start"], t["end"]) for t in suoi]
    media = np.mean(vettori, axis=0)  # impronta di riferimento = media dei vettori
    impronte_medie[nome] = media
    np.save(cartella_out / f"{nome}.npy", media)

    # controllo qualità: quanto ogni turno somiglia alla media?
    # un valore molto basso = turno sospetto (forse la diarizzazione ha sbagliato voce)
    sim = [somiglianza(v, media) for v in vettori]
    print(f"   somiglianza dei turni con la propria media: min {min(sim):.2f}, media {np.mean(sim):.2f}")


# ── 4. Le tre impronte sono distinte tra loro? ────────────────────────────
# Ci aspettiamo valori ALTI sulla diagonale (se stessi) e BASSI fuori.
print("\nSomiglianza tra impronte (più basso fuori diagonale = meglio):")
nomi = list(impronte_medie)
print("           " + "  ".join(f"{n:>9}" for n in nomi))
for a in nomi:
    righe = "  ".join(f"{somiglianza(impronte_medie[a], impronte_medie[b]):9.2f}" for b in nomi)
    print(f"{a:>10} {righe}")

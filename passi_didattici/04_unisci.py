"""
Passo 4: unisce TESTO (passo 2) e VOCI (passo 3).

Whisper ci ha dato:   [0.2 -> 10.2] "Questo programma può contenere..."
pyannote ci ha dato:  [0.0 -> 8.5]  SPEAKER_03
                      [8.5 -> 12.0] SPEAKER_02

Per ogni frase di Whisper guardiamo quali turni di pyannote la "toccano" e
scegliamo la voce che la copre PIÙ A LUNGO.

Output: data/trascrizioni/<id>_unito.json  con {start, end, speaker, text}

Uso:  .venv/bin/python 04_unisci.py
"""

import json
from pathlib import Path

ID_EPISODIO = "63134639027"

cartella = Path("data/trascrizioni")
frasi = json.load(open(cartella / f"{ID_EPISODIO}_whisper.json", encoding="utf-8"))
turni = json.load(open(cartella / f"{ID_EPISODIO}_speaker.json", encoding="utf-8"))


def sovrapposizione(a_inizio, a_fine, b_inizio, b_fine):
    """Quanti secondi hanno in comune due intervalli (0 se non si toccano)."""
    return max(0.0, min(a_fine, b_fine) - max(a_inizio, b_inizio))


unito = []
for frase in frasi:
    # secondi di sovrapposizione tra questa frase e ogni voce
    secondi_per_voce = {}
    for turno in turni:
        sec = sovrapposizione(frase["start"], frase["end"], turno["start"], turno["end"])
        if sec > 0:
            secondi_per_voce[turno["speaker"]] = secondi_per_voce.get(turno["speaker"], 0) + sec

    # la voce vincitrice è quella con più secondi; se nessuna copre la frase: "?"
    voce = max(secondi_per_voce, key=secondi_per_voce.get) if secondi_per_voce else "?"
    unito.append({"start": frase["start"], "end": frase["end"], "speaker": voce, "text": frase["text"]})

with open(cartella / f"{ID_EPISODIO}_unito.json", "w", encoding="utf-8") as f:
    json.dump(unito, f, ensure_ascii=False, indent=2)

# ── Anteprima: per ogni voce, le sue prime 4 frasi più lunghe ──────────────
# (serve a noi per capire a orecchio/occhio "chi è chi")
print(f"{len(unito)} frasi salvate\n")
for voce in sorted({u["speaker"] for u in unito}):
    sue = [u for u in unito if u["speaker"] == voce]
    print(f"── {voce}  ({len(sue)} frasi)")
    for u in sorted(sue, key=lambda x: -len(x["text"]))[:3]:
        print(f"   [{u['start']:6.1f}] {u['text'][:110]}")
    print()

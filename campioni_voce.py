"""
Estrae dagli episodi i campioni migliori della voce di una persona, da usare come
riferimento per la clonazione vocale (es. in VoiceStudio).

Un buon campione (la pulizia = quanto la voce emerge sul fondo: musica/rumore la abbassano):
  - è lungo 10-20 secondi, parlato di seguito da UNA sola persona (nessuno sopra);
  - ha una trascrizione affidabile (la clonazione vuole audio + testo);
  - ha una voce molto simile all'impronta vocale media della persona.

Uso:  .venv/bin/python campioni_voce.py [persona] [quanti]     (default: cruciani 3)
Scrive data/campioni_voce/<persona>_1.wav (+ .txt con la trascrizione), ecc.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import zanzara

PERSONA = sys.argv[1] if len(sys.argv) > 1 else "cruciani"
QUANTI = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DURATA_MIN, DURATA_MAX, DURATA_IDEALE = 10.0, 20.0, 14.0
PAROLE_AL_SECONDO = (1.8, 4.0)     # fuori da questo intervallo i tempi di Whisper non sono affidabili
CANDIDATI_PER_EPISODIO = 4         # quanti ne misuriamo davvero con la voce
USCITA = Path("data/campioni_voce")


def ha_loop(testo, lunghezza=4, volte=2):
    parole = testo.lower().split()
    return any(n >= volte for n in Counter(tuple(parole[i:i + lunghezza])
                                           for i in range(len(parole) - lunghezza + 1)).values())


def candidati_episodio(cartella):
    """Gruppi di frasi consecutive di PERSONA, senza altri che parlano sopra."""
    unito = json.load(open(cartella / "unito.json", encoding="utf-8"))
    turni = json.load(open(cartella / "turni.json", encoding="utf-8"))
    nomi = json.load(open(cartella / "nomi.json", encoding="utf-8"))
    # intervalli in cui parla qualcun altro (ignoriamo i micro-turni da meno di 0.3 s)
    altri = [(t["start"], t["end"]) for t in turni
             if nomi.get(t["speaker"]) != PERSONA and t["end"] - t["start"] >= 0.3]

    def libero(a, b):
        return not any(inizio < b and fine > a for inizio, fine in altri)

    candidati, gruppo = [], []

    def chiudi():
        if gruppo and DURATA_MIN <= gruppo[-1]["end"] - gruppo[0]["start"] <= DURATA_MAX:
            candidati.append(list(gruppo))

    for f in unito:
        durata = f["end"] - f["start"]
        parole = len(f["text"].split())
        valida = (f["persona"] == PERSONA and 0.5 <= durata <= 12
                  and PAROLE_AL_SECONDO[0] <= parole / durata <= PAROLE_AL_SECONDO[1]
                  and libero(f["start"], f["end"]))
        if valida and gruppo and f["start"] - gruppo[-1]["end"] <= 1.0:
            gruppo.append(f)
        else:
            if gruppo:
                chiudi()
            gruppo.clear()
            if valida:
                gruppo.append(f)
        # se il gruppo è già abbastanza lungo, lo salviamo e ne ricominciamo uno
        if gruppo and gruppo[-1]["end"] - gruppo[0]["start"] >= DURATA_IDEALE:
            chiudi()
            gruppo.clear()
    chiudi()
    candidati = [g for g in candidati if not ha_loop(" ".join(f["text"] for f in g))]
    candidati.sort(key=lambda g: abs((g[-1]["end"] - g[0]["start"]) - DURATA_IDEALE))
    return candidati


def pulizia_db(audio, inizio, fine):
    """Quanto la voce emerge sul fondo, in dB: livello del parlato meno livello nelle pause.
    Con musica o rumore di sottofondo le pause non sono silenziose e il valore scende."""
    campioni = audio["waveform"][0, int(inizio * 16000):int(fine * 16000)].numpy()
    n = len(campioni) // 160 * 160
    blocchi = campioni[:n].reshape(-1, 160)                    # finestre da 10 ms
    livello = 20 * np.log10(np.sqrt((blocchi ** 2).mean(axis=1)) + 1e-6)
    return float(np.percentile(livello, 90) - np.percentile(livello, 10))


def taglia(mp3, inizio, fine, destinazione):
    """Ritaglia dall'mp3 un wav mono a 24 kHz (adatto ai modelli di clonazione)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{inizio:.2f}", "-t", f"{fine - inizio:.2f}",
                    "-i", str(mp3), "-ac", "1", "-ar", "24000", str(destinazione)], check=True)


if __name__ == "__main__":
    impronta = np.atleast_2d(np.load(zanzara.DIR_IMPRONTE / f"{PERSONA}.npy")).mean(axis=0)
    migliori = []   # (somiglianza, episodio, gruppo)

    for cartella in sorted(zanzara.DIR_EPISODI.glob("*")):
        try:
            gruppi = candidati_episodio(cartella)
        except FileNotFoundError:
            continue
        if not gruppi:
            continue
        wav = zanzara.prepara_wav(cartella)          # servono i campioni audio per misurare la voce
        audio = zanzara.carica_audio(wav)
        # prima scartiamo i candidati con fondo sporco: teniamo i più puliti dell'episodio
        gruppi = sorted(gruppi, key=lambda g: -pulizia_db(audio, g[0]["start"], g[-1]["end"]))
        for g in gruppi[:CANDIDATI_PER_EPISODIO]:
            vettore = zanzara.vettori_turni(audio, [{"start": g[0]["start"], "end": g[-1]["end"]}])[0]
            migliori.append((zanzara.somiglianza(vettore, impronta), cartella, g,
                             pulizia_db(audio, g[0]["start"], g[-1]["end"])))
        wav.unlink(missing_ok=True)                  # si ricrea quando serve
        print(f"{cartella.name}: {len(gruppi)} candidati")

    # la voce deve essere quella giusta (soglia), poi vince chi ha il fondo più pulito
    soglia = 0.90 if PERSONA == "cruciani" else 0.82
    migliori = sorted((m for m in migliori if m[0] >= soglia), key=lambda x: -x[3])
    USCITA.mkdir(parents=True, exist_ok=True)
    scelti, episodi_usati = [], set()
    for somiglianza, cartella, g, pulizia in migliori:
        if cartella.name in episodi_usati:           # un solo campione per episodio: più varietà
            continue
        episodi_usati.add(cartella.name)
        scelti.append((somiglianza, cartella, g, pulizia))
        if len(scelti) == QUANTI:
            break

    for n, (somiglianza, cartella, g, pulizia) in enumerate(scelti, start=1):
        nome = USCITA / f"{PERSONA}_{n}"
        inizio, fine = max(0, g[0]["start"] - 0.1), g[-1]["end"] + 0.1
        taglia(cartella / "audio.mp3", inizio, fine, nome.with_suffix(".wav"))
        testo = " ".join(f["text"].strip() for f in g)
        nome.with_suffix(".txt").write_text(testo + "\n", encoding="utf-8")
        print(f"\n[{n}] {nome.with_suffix('.wav')}  ({fine - inizio:.1f} s, somiglianza {somiglianza:.2f}, pulizia {pulizia:.0f} dB, "
              f"episodio {cartella.name} min {inizio / 60:.1f})\n    «{testo}»")

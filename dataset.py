"""
Costruisce i "documenti" per il RAG: i turni di parola di Cruciani.

Parte dai file unito.json (frasi di Whisper con la persona che parla) e:
  1. unisce le frasi consecutive della stessa persona in un unico TURNO
  2. tiene solo i turni di Cruciani abbastanza lunghi
  3. a ogni turno allega il CONTESTO: cosa è stato detto subito prima da qualcun altro
  4. scrive tutto in data/rag/turni_cruciani.jsonl (un turno per riga)

Uso:  .venv/bin/python dataset.py [persona]     (default cruciani; es. parenzo)
"""

import glob
import json
import sys
from collections import Counter
from pathlib import Path

PERSONA = sys.argv[1] if len(sys.argv) > 1 else "cruciani"   # es. "parenzo"
PAROLE_MIN = 15          # turni più corti ("sì", "certo", "ma dai") non sono utili da recuperare
PAROLE_MAX = 250         # turni lunghissimi vengono tagliati in pezzi (più preciso nel recupero)
PAUSA_MAX = 6.0          # secondi: se la stessa persona riprende dopo più di così, nuovo turno
CONTESTO_PAROLE = 60     # quante parole del turno precedente conservare come contesto

OUT = Path(f"data/rag/turni_{PERSONA}.jsonl")


def data_episodio(id_episodio):
    """Titolo dell'episodio (es. 'La Zanzara del 12 novembre 2026'), se lo troviamo."""
    try:
        from zanzara import episodi_recenti
        return {str(e["id"]): e["title"] for e in episodi_recenti(30)}.get(id_episodio)
    except Exception:  # niente rete o credenziali: andiamo avanti senza titolo
        return None


def unisci_in_turni(frasi):
    """Frasi consecutive della stessa persona (senza lunghe pause) -> un solo turno."""
    turni = []
    for f in frasi:
        ultimo = turni[-1] if turni else None
        if ultimo and ultimo["persona"] == f["persona"] and f["start"] - ultimo["fine"] <= PAUSA_MAX:
            ultimo["testo"] += " " + f["text"]
            ultimo["fine"] = f["end"]
        else:
            turni.append({"persona": f["persona"], "inizio": f["start"], "fine": f["end"], "testo": f["text"]})
    return turni


def spezza(testo, massimo):
    """Divide un testo troppo lungo in pezzi da massimo ~'massimo' parole."""
    parole = testo.split()
    return [" ".join(parole[i:i + massimo]) for i in range(0, len(parole), massimo)]


def ultime_parole(testo, quante):
    return " ".join(testo.split()[-quante:])


def ha_loop(testo, lunghezza=4, volte=3):
    """True se la stessa sequenza di 'lunghezza' parole compare 'volte' volte:
    è il tipico errore di Whisper ("io ero in farmacia io ero in farmacia ...")."""
    parole = testo.lower().split()
    sequenze = Counter(tuple(parole[i:i + lunghezza]) for i in range(len(parole) - lunghezza + 1))
    return any(n >= volte for n in sequenze.values())


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    documenti = []
    scartati_loop = 0

    for file in sorted(glob.glob("data/episodi/*/unito.json")):
        id_episodio = file.split("/")[2]
        titolo = data_episodio(id_episodio)
        turni = unisci_in_turni(json.load(open(file, encoding="utf-8")))

        for i, turno in enumerate(turni):
            if turno["persona"] != PERSONA or len(turno["testo"].split()) < PAROLE_MIN:
                continue
            # contesto = turno precedente di un'altra persona (se c'è)
            contesto = ultime_parole(turni[i - 1]["testo"], CONTESTO_PAROLE) if i > 0 else ""
            for pezzo in spezza(turno["testo"], PAROLE_MAX):
                if ha_loop(pezzo):
                    scartati_loop += 1
                    continue
                documenti.append({
                    "id": f"{id_episodio}_{int(turno['inizio'])}_{len(documenti)}",
                    "episodio": id_episodio,
                    "titolo": titolo,
                    "minuto": round(turno["inizio"] / 60, 1),
                    "contesto": contesto,
                    "testo": pezzo,
                })

    with open(OUT, "w", encoding="utf-8") as f:
        for d in documenti:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Dataset per il fine-tuning (solo Cruciani): (cosa dice l'altro) -> (risposta di Cruciani)
    if PERSONA == "cruciani":
        with open(OUT.parent / "finetune.jsonl", "w", encoding="utf-8") as f:
            for d in documenti:
                if d["contesto"]:
                    f.write(json.dumps({"messages": [
                        {"role": "system", "content": "Sei Giuseppe Cruciani, conduttore de La Zanzara."},
                        {"role": "user", "content": d["contesto"]},
                        {"role": "assistant", "content": d["testo"]},
                    ]}, ensure_ascii=False) + "\n")

    # ── Statistiche ──────────────────────────────────────────────────────
    parole = [len(d["testo"].split()) for d in documenti]
    print(f"{len(documenti)} documenti salvati in {OUT}")
    print(f"parole: totale {sum(parole)}, media {sum(parole) / len(parole):.0f}, "
          f"min {min(parole)}, max {max(parole)}")
    print(f"scartati per loop di ripetizione: {scartati_loop}")
    print(f"con contesto: {sum(1 for d in documenti if d['contesto'])}")
    print("\n── 3 esempi a caso ──")
    for d in documenti[len(documenti) // 4::len(documenti) // 3][:3]:
        print(f"\n[{d['titolo'] or d['episodio']} · min {d['minuto']}]")
        print(f"  prima: «…{d['contesto'][-120:]}»")
        print(f"  {PERSONA.upper()}: «{d['testo'][:300]}»")

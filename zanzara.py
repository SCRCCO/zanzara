"""
Pipeline completa La Zanzara: episodio -> trascrizione con chi parla.

Due comandi:

  1) elabora: scarica un episodio, lo trascrive, capisce chi parla e prova a dare
     i nomi usando le impronte vocali già salvate.
        .venv/bin/python zanzara.py elabora              # ultimo episodio
        .venv/bin/python zanzara.py elabora --ultimi 10  # gli ultimi 10 (già fatti = saltati)

  2) nomi: gli dici tu chi è chi (dopo aver ascoltato i campioni) e lui salva le
     impronte vocali, poi riassegna i nomi all'episodio.
        .venv/bin/python zanzara.py nomi 63134639027 SPEAKER_01=cruciani SPEAKER_03=clizia

Ogni passo salva il risultato su disco (data/episodi/<id>/): se rilanci, i passi
già fatti vengono saltati.
"""

import argparse
import shutil
import hashlib
import json
import os
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

# ── Impostazioni ──────────────────────────────────────────────────────────
FEED_ID = 1068978                  # La Zanzara su Podcast Index
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"   # GPU del Mac
MODELLO_WHISPER = "mlx-community/whisper-large-v3-mlx"   # Whisper large-v3 sulla GPU del Mac
DURATA_MIN_TURNO = 3.0             # secondi: turni più corti non bastano per un'impronta
SOGLIA_NOME = 0.65                 # somiglianza minima per dare un nome a una voce
MARGINE_NOME = 0.05                # distacco minimo dalla seconda persona più simile
SOGLIA_OUTLIER = 0.60             # turni meno simili di così alla media vengono scartati
DIR_EPISODI = BASE / "data/episodi"
DIR_IMPRONTE = BASE / "data/impronte"


# ═════════════════════════════ 1. SCARICA ════════════════════════════════
def episodi_recenti(quanti):
    """Metadati degli ultimi 'quanti' episodi, dal più recente (senza duplicati)."""
    chiave, segreto = os.environ["PODCASTINDEX_API_KEY"], os.environ["PODCASTINDEX_API_SECRET"]
    ora = str(int(time.time()))
    intestazioni = {
        "User-Agent": "zanzara-llm-didattico/1.0",
        "X-Auth-Key": chiave,
        "X-Auth-Date": ora,
        "Authorization": hashlib.sha1((chiave + segreto + ora).encode()).hexdigest(),
    }
    r = requests.get(
        "https://api.podcastindex.org/api/1.0/episodes/byfeedid",
        params={"id": FEED_ID, "max": quanti + 5}, headers=intestazioni, timeout=20,
    )
    r.raise_for_status()

    # Podcast Index a volte restituisce lo stesso episodio due volte: togliamo i doppioni
    visti, episodi = set(), []
    for e in r.json()["items"]:
        if e["id"] not in visti:
            visti.add(e["id"])
            episodi.append(e)
    return episodi[:quanti]


def scarica(episodio, cartella):
    mp3 = cartella / "audio.mp3"
    if not mp3.exists():
        print(f"Scarico: {episodio['title']}")
        with requests.get(episodio["enclosureUrl"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(mp3, "wb") as f:
                for pezzo in r.iter_content(1024 * 1024):
                    f.write(pezzo)

    return prepara_wav(cartella)


def prepara_wav(cartella):
    """WAV mono 16 kHz (il formato che vogliono Whisper e pyannote), creato dall'mp3 se manca.
    Pesa ~170 MB: si può cancellare dopo l'elaborazione, verrà ricreato al bisogno."""
    wav = cartella / "audio.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(cartella / "audio.mp3"),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def carica_audio(wav):
    """WAV -> {'waveform': tensore (1, campioni), 'sample_rate': 16000}, formato di pyannote."""
    with wave.open(str(wav), "rb") as w:
        frequenza, byte = w.getframerate(), w.readframes(w.getnframes())
    campioni = np.frombuffer(byte, dtype=np.int16).astype(np.float32) / 32768.0
    return {"waveform": torch.from_numpy(campioni).unsqueeze(0), "sample_rate": frequenza}


# ═════════════════════════ 2. TESTO (Whisper) ════════════════════════════
def trascrivi(wav, cartella):
    file = cartella / "whisper.json"
    if file.exists():
        return json.load(open(file, encoding="utf-8"))

    import mlx_whisper
    print("Trascrivo con Whisper large-v3 (~9 minuti per episodio)...")
    # condition_on_previous_text=False: ogni pezzo è trascritto per conto suo, così Whisper
    # non entra nei loop di ripetizione ("io ero in farmacia io ero in farmacia...")
    risultato = mlx_whisper.transcribe(str(wav), path_or_hf_repo=MODELLO_WHISPER, language="it",
                                       condition_on_previous_text=False, verbose=None)
    frasi = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
             for s in risultato["segments"]]
    json.dump(frasi, open(file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return frasi


# ═══════════════════════ 3. VOCI (pyannote) ══════════════════════════════
def diarizza(audio, cartella):
    file = cartella / "turni.json"
    if file.exists():
        return json.load(open(file, encoding="utf-8"))

    from pyannote.audio import Pipeline
    print(f"Analizzo le voci con pyannote su {DEVICE}...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ["HF_KEY"])
    pipeline.to(torch.device(DEVICE))
    risultato = pipeline(audio)
    annotazione = getattr(risultato, "speaker_diarization", risultato)

    turni = [{"start": round(i.start, 2), "end": round(i.end, 2), "speaker": voce}
             for i, _, voce in annotazione.itertracks(yield_label=True)]
    json.dump(turni, open(file, "w", encoding="utf-8"), indent=1)
    return turni


# ═══════════════════ 4. IMPRONTE VOCALI E NOMI ═══════════════════════════
_estrattore = None


def vettori_turni(audio, turni):
    """Un vettore (impronta) per ogni turno lungo abbastanza. Ritorna array (N, dimensione)."""
    global _estrattore
    from pyannote.audio import Inference, Model
    from pyannote.core import Segment
    if _estrattore is None:
        modello = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=os.environ["HF_KEY"])
        _estrattore = Inference(modello, window="whole")
    lunghi = [t for t in turni if t["end"] - t["start"] >= DURATA_MIN_TURNO]
    if not lunghi:
        return np.empty((0, 256))
    return np.array([_estrattore.crop(audio, Segment(t["start"], t["end"])) for t in lunghi])


def somiglianza(a, b):
    """Somiglianza del coseno tra due vettori: 1 = stessa voce, ~0.5 = voci diverse."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def carica_impronte():
    """{nome: array (N, dimensione)} dai file in data/impronte/."""
    return {f.stem: np.atleast_2d(np.load(f)) for f in DIR_IMPRONTE.glob("*.npy")}


def assegna_nomi(audio, turni):
    """Per ogni voce anonima (SPEAKER_xx) cerca l'impronta nota più simile."""
    note = carica_impronte()
    nomi = {}
    for voce in sorted({t["speaker"] for t in turni}):
        vettori = vettori_turni(audio, [t for t in turni if t["speaker"] == voce])
        if len(vettori) == 0 or not note:
            nomi[voce] = None
            continue
        media = vettori.mean(axis=0)
        punteggi = {nome: somiglianza(media, v.mean(axis=0)) for nome, v in note.items()}
        classifica = sorted(punteggi, key=punteggi.get, reverse=True)
        migliore = classifica[0]
        secondo = punteggi[classifica[1]] if len(classifica) > 1 else 0.0
        # serve una somiglianza alta E un distacco netto dal secondo: se due persone
        # fanno punteggi quasi uguali (es. 0.66 e 0.64) la voce è probabilmente di un terzo
        sicura = punteggi[migliore] >= SOGLIA_NOME and punteggi[migliore] - secondo >= MARGINE_NOME
        nomi[voce] = migliore if sicura else None
        dettaglio = "  ".join(f"{n}={p:.2f}" for n, p in sorted(punteggi.items()))
        print(f"   {voce}: {nomi[voce] or 'sconosciuto':<10} ({dettaglio})")
    return nomi


def salva_impronte(audio, turni, assegnazioni):
    """assegnazioni = {'SPEAKER_01': 'cruciani', ...}. Aggiunge vettori alle impronte note."""
    DIR_IMPRONTE.mkdir(parents=True, exist_ok=True)
    for voce, nome in assegnazioni.items():
        vettori = vettori_turni(audio, [t for t in turni if t["speaker"] == voce])
        media = vettori.mean(axis=0)
        # scarta i turni "strani" (probabili errori di diarizzazione), se ne restano abbastanza
        buoni = np.array([v for v in vettori if somiglianza(v, media) >= SOGLIA_OUTLIER])
        if len(buoni) >= 3:
            vettori = buoni
        file = DIR_IMPRONTE / f"{nome}.npy"
        if file.exists():
            vettori = np.vstack([np.atleast_2d(np.load(file)), vettori])
        np.save(file, vettori)
        print(f"   {nome}: {len(vettori)} vettori totali salvati")


# ═══════════════════ 5. UNISCI TESTO + VOCI + NOMI ═══════════════════════
def unisci(frasi, turni, nomi, cartella):
    """A ogni frase di Whisper assegna la voce che la copre più a lungo."""
    unito = []
    for frase in frasi:
        secondi = {}
        for t in turni:
            comune = min(frase["end"], t["end"]) - max(frase["start"], t["start"])
            if comune > 0:
                secondi[t["speaker"]] = secondi.get(t["speaker"], 0) + comune
        voce = max(secondi, key=secondi.get) if secondi else "?"
        unito.append({**frase, "speaker": voce, "persona": nomi.get(voce) or "sconosciuto"})
    json.dump(unito, open(cartella / "unito.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return unito


def ritaglia_campioni(wav, turni, nomi, cartella):
    """Per ogni voce senza nome salva 2 clip (max 25 s) da ascoltare."""
    (cartella / "campioni").mkdir(exist_ok=True)
    for voce in sorted({t["speaker"] for t in turni}):
        suoi = sorted((t for t in turni if t["speaker"] == voce), key=lambda t: t["start"] - t["end"])
        if nomi.get(voce) or sum(t["end"] - t["start"] for t in suoi) < 20:
            continue  # già riconosciuta, oppure troppo poco parlato per valere un clip
        for n, t in enumerate(suoi[:2], start=1):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t["start"]),
                            "-t", str(min(t["end"] - t["start"], 25)), "-i", str(wav),
                            str(cartella / "campioni" / f"{voce}_{n}.wav")], check=True)


def riepilogo(unito):
    secondi = {}
    for u in unito:
        secondi[u["persona"]] = secondi.get(u["persona"], 0) + u["end"] - u["start"]
    print("\nParlato per persona:")
    for p, s in sorted(secondi.items(), key=lambda x: -x[1]):
        print(f"   {p:<12} {s / 60:5.1f} min")


# ═══════════════════════════════ COMANDI ═════════════════════════════════
def cmd_elabora(args):
    """Elabora gli ultimi N episodi; se uno fallisce passa al successivo."""
    episodi = episodi_recenti(args.ultimi)
    print(f"{len(episodi)} episodi da elaborare")
    for numero, episodio in enumerate(episodi, start=1):
        print(f"\n=== [{numero}/{len(episodi)}] {time.strftime('%H:%M')} ===")
        try:
            elabora_episodio(episodio)
            if args.pulisci:  # wav e campioni si ricreano: tolgono ~250 MB a episodio
                cartella = DIR_EPISODI / str(episodio["id"])
                (cartella / "audio.wav").unlink(missing_ok=True)
                shutil.rmtree(cartella / "campioni", ignore_errors=True)
        except Exception as errore:  # un episodio rotto non deve fermare gli altri
            print(f"ERRORE su {episodio['id']}: {errore}")


def elabora_episodio(episodio):
    cartella = DIR_EPISODI / str(episodio["id"])
    cartella.mkdir(parents=True, exist_ok=True)
    print(f"Episodio {episodio['id']}: {episodio['title']}")

    wav = scarica(episodio, cartella)
    audio = carica_audio(wav)
    frasi = trascrivi(wav, cartella)
    turni = diarizza(audio, cartella)

    # I nomi si calcolano una volta sola per episodio (è il passo lento) e si salvano
    file_nomi = cartella / "nomi.json"
    primo_giro = not file_nomi.exists()
    if primo_giro:
        print("Riconoscimento voci:")
        json.dump(assegna_nomi(audio, turni), open(file_nomi, "w", encoding="utf-8"), indent=1)
    nomi = json.load(open(file_nomi, encoding="utf-8"))

    unito = unisci(frasi, turni, nomi, cartella)
    if primo_giro:
        ritaglia_campioni(wav, turni, nomi, cartella)
    riepilogo(unito)
    print(f"\nRisultato in {cartella}/unito.json")
    if any(v is None for v in nomi.values()):
        print(f"Voci senza nome: ascolta {cartella}/campioni/ e poi lancia:\n"
              f"   .venv/bin/python zanzara.py nomi {episodio['id']} SPEAKER_xx=nome ...")


def cmd_nomi(args):
    cartella = DIR_EPISODI / args.id
    assegnazioni = dict(coppia.split("=") for coppia in args.assegnazioni)
    audio = carica_audio(prepara_wav(cartella))
    turni = json.load(open(cartella / "turni.json", encoding="utf-8"))
    frasi = json.load(open(cartella / "whisper.json", encoding="utf-8"))

    print("Salvo le impronte:")
    salva_impronte(audio, turni, assegnazioni)
    print("Riassegno i nomi:")
    nomi = assegna_nomi(audio, turni)
    nomi.update(assegnazioni)  # quello che hai detto tu vale sempre
    json.dump(nomi, open(cartella / "nomi.json", "w", encoding="utf-8"), indent=1)
    riepilogo(unisci(frasi, turni, nomi, cartella))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sotto = parser.add_subparsers(dest="comando", required=True)

    p = sotto.add_parser("elabora")
    p.add_argument("--ultimi", type=int, default=1, help="quanti episodi recenti elaborare (default 1)")
    p.add_argument("--pulisci", action="store_true", help="a fine episodio cancella wav e campioni (risparmia disco)")
    p.set_defaults(func=cmd_elabora)

    p = sotto.add_parser("nomi")
    p.add_argument("id", help="id episodio (nome della cartella in data/episodi/)")
    p.add_argument("assegnazioni", nargs="+", help="es. SPEAKER_01=cruciani")
    p.set_defaults(func=cmd_nomi)

    argomenti = parser.parse_args()
    argomenti.func(argomenti)

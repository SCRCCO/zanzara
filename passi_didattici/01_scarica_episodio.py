"""
Passo 1: scarica l'ultimo episodio de La Zanzara.

Cosa fa:
  1. Si autentica su Podcast Index (chiave + segreto dal file .env)
  2. Chiede gli ultimi episodi del feed de La Zanzara
  3. Scarica l'audio dell'ultimo in data/audio/

Uso:  .venv/bin/python 01_scarica_episodio.py
"""

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

FEED_ID = 1068978  # La Zanzara, Radio 24 (trovato cercando su Podcast Index)
CARTELLA_AUDIO = Path("data/audio")

load_dotenv()  # legge le variabili dal file .env
API_KEY = os.environ["PODCASTINDEX_API_KEY"]
API_SECRET = os.environ["PODCASTINDEX_API_SECRET"]


def intestazioni_auth():
    """Podcast Index vuole: chiave, timestamp e un hash sha1(chiave+segreto+timestamp)."""
    ora = str(int(time.time()))
    firma = hashlib.sha1((API_KEY + API_SECRET + ora).encode()).hexdigest()
    return {
        "User-Agent": "zanzara-llm-didattico/1.0",
        "X-Auth-Key": API_KEY,
        "X-Auth-Date": ora,
        "Authorization": firma,
    }


def ultimi_episodi(quanti=3):
    url = "https://api.podcastindex.org/api/1.0/episodes/byfeedid"
    risposta = requests.get(
        url, params={"id": FEED_ID, "max": quanti}, headers=intestazioni_auth(), timeout=20
    )
    risposta.raise_for_status()
    return risposta.json()["items"]


def scarica(url, destinazione):
    """Scarica il file a pezzi (stream), così non carica tutto in memoria."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(destinazione, "wb") as f:
            for pezzo in r.iter_content(chunk_size=1024 * 1024):
                f.write(pezzo)


if __name__ == "__main__":
    CARTELLA_AUDIO.mkdir(parents=True, exist_ok=True)

    episodi = ultimi_episodi()
    for e in episodi:
        data = datetime.fromtimestamp(e["datePublished"]).strftime("%Y-%m-%d")
        print(f"{data} | {e['duration'] // 60:>3} min | {e['title']}")

    ultimo = episodi[0]
    file_audio = CARTELLA_AUDIO / f"{ultimo['id']}.mp3"
    if file_audio.exists():
        print(f"\nGià scaricato: {file_audio}")
    else:
        print(f"\nScarico: {ultimo['title']}")
        scarica(ultimo["enclosureUrl"], file_audio)
        print(f"Salvato in {file_audio} ({file_audio.stat().st_size / 1e6:.1f} MB)")

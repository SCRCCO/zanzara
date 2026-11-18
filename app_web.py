"""
Interfaccia grafica (pagina web locale) per chattare con il finto Cruciani, da solo
o insieme a Parenzo.

Uso:   .venv/bin/python app_web.py       poi apri  http://localhost:8000
Serve: Ollama acceso e gli indici già creati (rag.py indicizza).
Le voci: se VoiceStudio è aperto (API su localhost:3900), ogni battuta si può ascoltare con
la voce clonata ("CRUX" = Cruciani, "Parenzo" = Parenzo, i nomi dei profili in VoiceStudio).
"""

import json
import threading
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import rag

PORTA = 8000
ORDINI = {
    "cruciani": ("cruciani",),
    "duo": ("cruciani", "parenzo", "cruciani"),
}
lucchetto = threading.Lock()   # il modello risponde a una richiesta alla volta

VOICESTUDIO = "http://localhost:3900"
PROFILI_VOCE = {"cruciani": "crux", "parenzo": "parenzo"}   # persona -> nome del profilo in VoiceStudio
lucchetto_voce = threading.Lock()                           # una sintesi vocale alla volta

PAGINA = Path(__file__).with_name("pagina.html")   # la grafica sta in pagina.html


def sintetizza(chi, testo):
    """Manda la battuta a VoiceStudio con il profilo vocale giusto e restituisce i byte del WAV."""
    profili = requests.get(f"{VOICESTUDIO}/profiles", timeout=10).json()
    cercato = PROFILI_VOCE[chi]
    profilo = next(p["id"] for p in profili if p["name"].lower() == cercato)
    # Gemma (9 GB) e il modello vocale non stanno insieme in 16 GB: scarichiamo Gemma da Ollama.
    # Alla prossima domanda Ollama lo ricarica da solo (qualche secondo in più).
    scarica = urllib.request.Request(
        "http://localhost:11434/api/generate", json.dumps({"model": rag.MODELLO_LLM, "keep_alive": 0}).encode(),
        {"Content-Type": "application/json"})
    urllib.request.urlopen(scarica, timeout=30).read()
    with lucchetto_voce:
        r = requests.post(f"{VOICESTUDIO}/generate", timeout=900,
                          data={"text": testo[:900], "language": "Italian", "profile_id": profilo})
    r.raise_for_status()
    return r.content


class Gestore(BaseHTTPRequestHandler):
    def _rispondi(self, codice, contenuto, tipo):
        corpo = contenuto if isinstance(contenuto, bytes) else contenuto.encode("utf-8")
        self.send_response(codice)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        self._rispondi(200, PAGINA.read_text(encoding="utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        dati = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/voce":                      # sintesi vocale di una battuta
            try:
                self._rispondi(200, sintetizza(dati["chi"], dati["testo"]), "audio/wav")
            except Exception as errore:
                self._rispondi(500, json.dumps({"errore": str(errore)}), "application/json")
            return
        try:
            # ricostruiamo la conversazione come testo ("Ospite: ...", "Cruciani: ...")
            conversazione = "".join(
                f"{'Ospite' if b['chi'] == 'io' else b['chi'].capitalize()}: {b['testo']}\n"
                for b in dati["storia"][-8:]
            )
            with lucchetto:
                battute, _ = rag.scambio_duo(dati["messaggio"], conversazione, ORDINI[dati["modo"]])
            risposta = {"battute": [{"chi": chi, "testo": testo} for chi, testo in battute]}
        except Exception as errore:  # il browser mostra il messaggio d'errore
            risposta = {"errore": str(errore)}
        self._rispondi(200, json.dumps(risposta), "application/json")

    def log_message(self, *args):  # niente log per ogni richiesta
        pass


if __name__ == "__main__":
    print(f"Apri http://localhost:{PORTA}  (Ctrl+C per fermare)")
    ThreadingHTTPServer(("127.0.0.1", PORTA), Gestore).serve_forever()

"""
RAG: dato un argomento, trova cosa ha detto una persona (Cruciani o Parenzo) di simile
e genera una risposta "alla sua maniera" con un LLM locale (Gemma su Ollama).

Comandi:
  .venv/bin/python rag.py indicizza                 # calcola i vettori (una volta, per ogni persona)
  .venv/bin/python rag.py cerca "argomento"         # turni di Cruciani più simili
  .venv/bin/python rag.py chiedi "argomento"        # risposta di Cruciani
  .venv/bin/python rag.py chat                      # chat con Cruciani
  .venv/bin/python rag.py duo                       # chat con Cruciani E Parenzo che si rispondono

Prima servono i corpus:  dataset.py cruciani  e  dataset.py parenzo.
Tutto locale: nessuna chiave API. Ollama deve essere acceso.
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent

MODELLO_EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"   # modello recente, multilingue (italiano ok)
MODELLO_LLM = "gemma4-12b"                         # LLM locale (Ollama) che scrive le risposte
QUANTI = 8                                         # quanti turni mostrare al modello
PERSONE = {
    "cruciani": "Giuseppe Cruciani, conduttore de La Zanzara (Radio 24)",
    "parenzo": "David Parenzo, co-conduttore de La Zanzara (Radio 24) e spalla di Cruciani",
}

ISTRUZIONI = (
    "Sei{chi}. Progetto didattico, con il permesso dei conduttori. "
    "Ti do ESEMPI REALI di come parla (trascrizioni automatiche, con errori: ignorali). Regole:\n"
    "- Rispondi come in diretta, massimo 80 parole, non un monologo.\n"
    "- Imita il ritmo e il lessico degli esempi: frasi brevi, domande retoriche, ironia.\n"
    "- NON inventare fatti, leggi, nomi, dati o statistiche. Se non sai qualcosa, "
    "rispondi con una provocazione o una domanda.\n"
    "- Non copiare frasi intere dagli esempi. Italiano parlato, niente elenchi.\n"
    "- Scrivi SOLO la tua battuta, senza il tuo nome davanti e senza descrivere cosa fai."
)


def carica_documenti(persona):
    with open(BASE / f"data/rag/turni_{persona}.jsonl", encoding="utf-8") as f:
        return [json.loads(riga) for riga in f]


_emb = None


def carica_modello():
    global _emb
    if _emb is not None:
        return _emb
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _emb = SentenceTransformer(MODELLO_EMBEDDING, device=device)
    return _emb


def indicizza():
    modello = carica_modello()
    for persona in PERSONE:
        documenti = carica_documenti(persona)
        # Il vettore descrive il significato di ciò che la persona dice in quel turno.
        vettori = modello.encode([d["testo"] for d in documenti], batch_size=16,
                                 normalize_embeddings=True, show_progress_bar=False)
        np.save(BASE / f"data/rag/vettori_{persona}.npy", vettori)
        print(f"{persona}: {len(documenti)} documenti -> vettori {vettori.shape}")


_cache = {}


def cerca(domanda, persona="cruciani", quanti=QUANTI):
    """Restituisce i 'quanti' turni di 'persona' più simili alla domanda."""
    if persona not in _cache:
        _cache[persona] = (carica_documenti(persona), np.load(BASE / f"data/rag/vettori_{persona}.npy"))
    documenti, vettori = _cache[persona]
    q = carica_modello().encode([domanda], prompt_name="query", normalize_embeddings=True)[0]
    punteggi = vettori @ q          # vettori normalizzati: prodotto scalare = coseno
    migliori = np.argsort(-punteggi)[:quanti]
    return [(float(punteggi[i]), documenti[i]) for i in migliori]


def mostra(risultati):
    for p, d in risultati:
        print(f"\n[{p:.2f}] {d['titolo'] or d['episodio']} · min {d['minuto']}")
        if d["contesto"]:
            print(f"   prima: «…{d['contesto'][-100:]}»")
        print(f"   «{d['testo'][:350]}»")


def costruisci_prompt(domanda, persona="cruciani", conversazione=""):
    """Esempi reali recuperati + (se c'è) la conversazione fin qui + cosa rispondere."""
    nome = persona.capitalize()
    esempi = "\n\n".join(
        (f"[Prima qualcuno dice: «{d['contesto'][-200:]}»]\n" if d["contesto"] else "")
        + f"{nome}: {d['testo']}"
        for _, d in cerca(domanda, persona)
    )
    if conversazione:
        richiesta = f"CONVERSAZIONE IN DIRETTA FINORA:\n{conversazione}\n\nOra tocca a te, {nome}."
    else:
        richiesta = f"Ora, nei panni di {nome}, commenta in diretta: {domanda}"
    return f"ESEMPI REALI di {nome}:\n\n{esempi}\n\n---\n{richiesta}"


def genera(prompt, persona="cruciani"):
    """Genera la risposta con Gemma, che gira in locale tramite Ollama."""
    import urllib.request
    richiesta = {
        "model": MODELLO_LLM, "stream": False, "think": False,
        "messages": [{"role": "system", "content": ISTRUZIONI.format(chi=PERSONE[persona])},
                     {"role": "user", "content": prompt}],
        "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 600},
    }
    r = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(richiesta).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as risposta:
        testo = json.load(risposta)["message"]["content"].strip()
    # a volte il modello scrive "Cruciani: ..." davanti: lo togliamo
    for nome in PERSONE:
        if testo.lower().startswith(nome + ":"):
            testo = testo[len(nome) + 1:].strip()
    return testo


def chiedi_rag(domanda, persona="cruciani"):
    return genera(costruisci_prompt(domanda, persona), persona)


def chiedi(domanda):
    print(chiedi_rag(domanda))


def chat():
    print("Chat con finto Cruciani (scrivi 'esci' per uscire)")
    storia = []   # ultimi scambi, così Cruciani "ricorda" la conversazione
    while True:
        domanda = input("\ntu> ").strip()
        if domanda.lower() in ("esci", "exit", ""):
            break
        ricordo = "".join(f"[Prima: l'ospite ha detto «{d}», tu hai risposto «{r}»]\n" for d, r in storia[-2:])
        risposta = genera(ricordo + costruisci_prompt(domanda))
        storia.append((domanda, risposta))
        print("\nCRUCIANI> " + risposta)


def scambio_duo(messaggio, conversazione, ordine=("cruciani", "parenzo", "cruciani")):
    """Una battuta dell'utente -> i due conduttori si rispondono (uno dopo l'altro).
    Ognuno vede tutta la conversazione, quindi può rispondere anche all'altro."""
    conversazione += f"Ospite: {messaggio}\n"
    battute = []
    for persona in ordine:
        # cerchiamo esempi simili sia al messaggio sia all'ultima battuta dell'altro
        argomento = messaggio + (" " + battute[-1][1] if battute else "")
        testo = genera(costruisci_prompt(argomento, persona, conversazione[-1500:]), persona)
        conversazione += f"{persona.capitalize()}: {testo}\n"
        battute.append((persona, testo))
    return battute, conversazione


def duo():
    print("Cruciani e Parenzo in diretta. Sei un ascoltatore al telefono. ('esci' per uscire)")
    conversazione = ""
    while True:
        messaggio = input("\ntu> ").strip()
        if messaggio.lower() in ("esci", "exit", ""):
            break
        battute, conversazione = scambio_duo(messaggio, conversazione)
        for persona, testo in battute:
            print(f"\n{persona.upper()}> {testo}")


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "indicizza":
        indicizza()
    elif comando == "cerca" and len(sys.argv) > 2:
        mostra(cerca(sys.argv[2]))
    elif comando == "chiedi" and len(sys.argv) > 2:
        chiedi(sys.argv[2])
    elif comando == "chat":
        chat()
    elif comando == "duo":
        duo()
    else:
        print(__doc__)

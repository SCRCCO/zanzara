"""
Fine-tuning (LoRA) di un modello su come parla Cruciani, con mlx-lm (GPU del Mac).
Separato dal RAG, così si possono confrontare i due approcci.

Passi (in ordine):
  .venv/bin/python dataset.py                      # (ri)crea data/rag/finetune.jsonl dai turni
  .venv/bin/python finetune.py prepara             # divide in train/valid
  .venv/bin/python finetune.py allena [riprendi]   # LoRA (riprendi = dall'ultimo checkpoint dopo un crash): lungo, tienilo in carica e a coperchio aperto
  .venv/bin/python finetune.py prova "frase"       # risponde il modello addestrato
  .venv/bin/python finetune.py confronta "frase"   # affianca: RAG (Gemma+esempi) vs fine-tuning

NON lanciare "allena" mentre gira "zanzara.py elabora": si contendono GPU e RAM (16 GB).

Nota: il Gemma GGUF di Ollama non si può addestrare; usiamo la stessa famiglia in
formato MLX (4 bit), che sta in 16 GB con LoRA + grad-checkpoint.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
SORGENTE = BASE / "data/rag/finetune.jsonl"
CARTELLA = BASE / "data/finetune"
ADAPTER = CARTELLA / "adapter"

MODELLO = "mlx-community/gemma-4-e2b-it-4bit"   # piccolo: veloce e stabile in 16 GB (il 12B-OptiQ-4bit è andato in crash Metal e ha dato testo senza senso)
ITERAZIONI = 800          # 1 iterazione = 1 esempio (batch 1): con ~2200 esempi sono circa mezza epoca; alza se servono
SYSTEM = "Sei Giuseppe Cruciani, conduttore de La Zanzara."
MAX_PAROLE_PROMPT = 60    # a un esempio serve solo la fine di ciò che ha detto l'altro


def prepara():
    """Divide gli esempi in train (90%) e valid (10%), in ordine casuale ma ripetibile."""
    esempi = [json.loads(r) for r in open(SORGENTE, encoding="utf-8")]
    for e in esempi:  # accorcia il contesto: meno token, allenamento più veloce
        e["messages"][1]["content"] = " ".join(e["messages"][1]["content"].split()[-MAX_PAROLE_PROMPT:])
    random.Random(42).shuffle(esempi)
    n_valid = max(10, len(esempi) // 10)
    CARTELLA.mkdir(parents=True, exist_ok=True)
    for nome, parte in (("train", esempi[n_valid:]), ("valid", esempi[:n_valid])):
        with open(CARTELLA / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for e in parte:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"train: {len(esempi) - n_valid} esempi, valid: {n_valid} in {CARTELLA}")


def allena():
    comando = [
        sys.executable, "-m", "mlx_lm", "lora", "--train",
        "--model", MODELLO, "--data", str(CARTELLA),
        "--adapter-path", str(ADAPTER),
        "--iters", str(ITERAZIONI), "--batch-size", "1",
        "--num-layers", "16", "--learning-rate", "1e-4", "--max-seq-length", "512",
        "--mask-prompt",            # la loss conta solo sulle parole di Cruciani, non su quelle dell'altro
        "--grad-checkpoint",        # meno memoria, un po' più lento
        "--steps-per-eval", "100", "--save-every", "100",
    ]
    if "riprendi" in sys.argv and (ADAPTER / "adapters.safetensors").exists():
        comando += ["--resume-adapter-file", str(ADAPTER / "adapters.safetensors")]  # riparte dall'ultimo checkpoint
    print(" ".join(comando))
    subprocess.run(comando, check=True)


_modello = None


def risposta_finetuned(frase):
    global _modello
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    if _modello is None:
        _modello = load(MODELLO, adapter_path=str(ADAPTER))
    modello, tokenizer = _modello
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": frase}],
        add_generation_prompt=True, tokenize=False)
    return generate(modello, tokenizer, prompt=prompt, max_tokens=300,
                    sampler=make_sampler(temp=0.8, top_p=0.95)).strip()


def confronta(frase):
    """Prima il RAG, poi il fine-tuning: i due modelli NON devono stare in memoria insieme
    (16 GB non bastano: Metal va in errore). Per questo il RAG gira in un processo a parte
    e alla fine scarichiamo Gemma da Ollama prima di caricare il modello addestrato."""
    import json
    import urllib.request
    rag = subprocess.run([sys.executable, str(BASE / "rag.py"), "chiedi", frase],
                         capture_output=True, text=True)       # il processo finisce -> libera memoria
    print("── RAG (Gemma + esempi recuperati) ──\n" + rag.stdout.strip())
    richiesta = urllib.request.Request(                       # keep_alive 0 = scarica subito il modello
        "http://localhost:11434/api/generate", json.dumps({"model": "gemma4-12b", "keep_alive": 0}).encode(),
        {"Content-Type": "application/json"})
    urllib.request.urlopen(richiesta, timeout=60).read()
    print("\n── FINE-TUNING (senza esempi) ──\n" + risposta_finetuned(frase))


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "prepara":
        prepara()
    elif comando == "allena":
        allena()
    elif comando == "prova" and len(sys.argv) > 2:
        print(risposta_finetuned(sys.argv[2]))
    elif comando == "confronta" and len(sys.argv) > 2:
        confronta(sys.argv[2])
    else:
        print(__doc__)

"""Aufruf: python nuextract3.py dokument.pdf felder.json"""
import json
import sys

import pymupdf
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

if len(sys.argv) != 3:
    sys.exit("Aufruf: python nuextract3.py dokument.pdf felder.json")

# Ausschließlich Seite 1 lesen, kein OCR.
with pymupdf.open(sys.argv[1]) as pdf:
    if len(pdf) == 0:
        sys.exit("Das PDF enthält keine Seiten.")
    text = pdf[0].get_text("text")
if not text.strip():
    sys.exit("Seite 1 enthält keinen auslesbaren Text.")

with open(sys.argv[2], encoding="utf-8") as f:
    felder = json.load(f)
if not isinstance(felder, dict):
    sys.exit("Die JSON-Vorlage muss ein Objekt mit Feldnamen enthalten.")

# Leere Werte der bisherigen Feldvorlage als String-Felder interpretieren.
schema = {name: typ if typ else "string" for name, typ in felder.items()}
modellname = "numind/NuExtract3"
prozessor = AutoProcessor.from_pretrained(modellname, trust_remote_code=True)
modell = AutoModelForImageTextToText.from_pretrained(
    modellname, dtype="auto", device_map="auto", trust_remote_code=True
).eval()

nachrichten = [{"role": "user", "content": [{"type": "text", "text": text}]}]
eingabe = prozessor.apply_chat_template(
    nachrichten, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    template=json.dumps(schema, ensure_ascii=False), enable_thinking=False,
).to(modell.device)

with torch.inference_mode():
    ausgabe = modell.generate(**eingabe, max_new_tokens=1024, do_sample=False)

antwort = prozessor.batch_decode(
    ausgabe[:, eingabe.input_ids.shape[1]:], skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()

try:
    daten = json.loads(antwort)
except json.JSONDecodeError:
    sys.exit("Das Modell hat kein gültiges JSON erzeugt. Modellausgabe:\n" + antwort[:1000])
print(json.dumps(daten, ensure_ascii=False, indent=2))

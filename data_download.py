from datasets import load_dataset
from tqdm import tqdm
import random, os

HEDEF = {"tr": 12_000_000, "en": 8_000_000}   # bayt -> 60/40 oranı

def topla(dil, hedef):
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{dil}",
                      split="train", streaming=True)
    parcalar, toplam = [], 0
    for makale in tqdm(ds, desc=dil):
        metin = makale["text"].strip()
        if len(metin) < 500:          # çok kısa taslak maddeleri ele
            continue
        parcalar.append(metin)
        toplam += len(metin.encode("utf-8"))
        if toplam >= hedef:
            break
    return parcalar

belgeler = topla("tr", HEDEF["tr"]) + topla("en", HEDEF["en"])

random.seed(42)
random.shuffle(belgeler)              # iki dili belge seviyesinde karıştır

os.makedirs("data", exist_ok=True)
with open("data/karisik.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(belgeler))

print("bitti:", os.path.getsize("data/karisik.txt") / 1e6, "MB")
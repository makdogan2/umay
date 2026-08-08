from collections import Counter
import regex as re

metin = open("data/mix.txt", encoding="utf-8").read()[:200_000]
DESEN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")

parcalar = DESEN.findall(metin)
frekans = Counter(parcalar)
kelimeler = {k: list(k) for k in frekans}

BIRLESME = 1000
birlesmeler = []

def sikisma():
    toplam = sum(len(kelimeler[k]) * n for k, n in frekans.items())
    return len(metin) / toplam

for adim in range(BIRLESME):
    ciftler = Counter()
    for kelime, sayi in frekans.items():
        p = kelimeler[kelime]
        for cift in zip(p, p[1:]):
            ciftler[cift] += sayi
    if not ciftler:
        break
    (a, b), _ = ciftler.most_common(1)[0]
    birlesmeler.append((a, b))

    for kelime in kelimeler:
        eski, yeni, i = kelimeler[kelime], [], 0
        while i < len(eski):
            if i < len(eski)-1 and eski[i] == a and eski[i+1] == b:
                yeni.append(a + b); i += 2
            else:
                yeni.append(eski[i]); i += 1
        kelimeler[kelime] = yeni

    if (adim + 1) % 100 == 0:
        son = [x + y for x, y in birlesmeler[-8:]]
        print(f"{adim+1:5d} birleşme | sıkışma {sikisma():.2f} | son: {son}")

# --- sonuç: bir cümle nasıl parçalanıyor ---
ornek = " Türkiye'de evlerimizden çıkarken gerçekleştirilen çalışmalar"
for parca in DESEN.findall(ornek):
    print(f"{parca!r:30} -> {kelimeler.get(parca, list(parca))}")
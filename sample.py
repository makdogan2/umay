import torch
from pathlib import Path
from model import Config, UMAY0

KOK = Path(__file__).parent
d = torch.load(KOK / "data" / "char.pt", weights_only=False)
stoi, itos = d["stoi"], d["itos"]

cfg = Config()
cfg.vocab_size = len(stoi)

model = UMAY0(cfg).to("cuda")
model.load_state_dict(torch.load(KOK / "umay0.pt", weights_only=False)["model"])
model.eval()

def uret(prompt, n=400, temperature=0.8, top_k=40):
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]],
                       dtype=torch.long, device="cuda")
    cikti = model.generate(idx, n, temperature, top_k)[0].tolist()
    return "".join(itos[i] for i in cikti)

for p in ["Türkiye, ", "Ankara şehri ", "The history of ", "Bilim insanları "]:
    print("=" * 60)
    print(uret(p))
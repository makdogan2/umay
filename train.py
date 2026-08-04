import torch, time
from pathlib import Path
from model import Config, UMAY0

KOK = Path(__file__).parent
d = torch.load(KOK / "data" / "char.pt", weights_only=False)
train_data, val_data = d["train"], d["val"]
itos = d["itos"]

cfg = Config()
cfg.vocab_size = len(d["stoi"])

batch_size = 64
max_iters  = 5000
eval_every = 500
lr         = 3e-4
device     = "cuda"

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+cfg.block_size].long() for i in ix])
    y = torch.stack([data[i+1:i+cfg.block_size+1].long() for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def olc():
    model.eval()
    out = {}
    for split in ("train", "val"):
        kayiplar = torch.zeros(50)
        for i in range(50):
            X, Y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
            kayiplar[i] = loss.item()
        out[split] = kayiplar.mean().item()
    model.train()
    return out

model = UMAY0(cfg).to(device)
print("parametre:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_every == 0:
        l = olc()
        print(f"adım {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time()-t0:.0f}s")

    X, Y = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, KOK / "umay0.pt")

baslangic = torch.zeros((1, 1), dtype=torch.long, device=device)
uretim = model.generate(baslangic, 500, temperature=0.8, top_k=40)[0].tolist()
print("\n--- UMAY konuşuyor ---")
print("".join(itos[i] for i in uretim))
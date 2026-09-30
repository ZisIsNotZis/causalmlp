import torch
from torch.nn import *
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

class Mul(Linear):
    """Causal MLP layer with cumulative mean."""
    def forward(self, x):
        return self.bias.addcmul(x[...,None], self.weight.T).cumsum(-2) / \
               torch.arange(1, x.size(-1)+1, device=x.device, dtype=x.dtype)[:,None]

# Reference model (frozen teacher)
ref = Sequential(
    Linear(16, 32),
    SiLU(),
    Linear(32, 10)
).cuda().requires_grad_(False).eval()

print("=" * 70)
print("Causal Hidden MLP: Required Hidden Units")
print("=" * 70)
print()

# Quick test with fewer iterations
hidden_sizes = [32, 64, 128, 256]

for h_size in hidden_sizes:
    print(f"h={h_size} ({h_size/32:.0f}x)...", end=" ", flush=True)
    
    m2 = Sequential(
        Linear(16, h_size),
        SiLU(),
        Mul(h_size, 10)
    ).cuda()
    
    opt = torch.optim.Adam(m2.parameters(), 0.01)
    
    for i in range(50):
        for _ in range(200):
            opt.zero_grad()
            x = torch.randn(256, 16, device='cuda')
            y2 = m2(x)
            t = ref(x).argmax(1)
            l2 = F.cross_entropy(y2.transpose(1,2), t[:,None].expand(-1, h_size), reduction='none')
            loss = l2.mean()
            loss.backward()
            opt.step()
    
    m2.eval()
    with torch.no_grad():
        x_test = torch.randn(10000, 16, device='cuda')
        t_test = ref(x_test).argmax(1)
        y2 = m2(x_test)
        
        per_exit = [(y2[:, i].argmax(1) == t_test).float().mean().item() for i in range(h_size)]
        best = max(per_exit)
        final = per_exit[-1]
        print(f"Best={best*100:.1f}%, Final={final*100:.1f}%")
    
    del m2
    torch.cuda.empty_cache()

print()
print("Note: model (standard MLP, h=32) reaches ~92-93% vs ref")
print("model2 (causal) needs more hidden units to approach this")

import torch
from torch.nn import *
class Mul(Linear):
	def forward(self, x):
		return self.bias.addcmul(x[...,None],self.weight.T).cumsum(-2)/torch.arange(1,x.size(-1)+1,device=x.device,dtype=x.dtype)[:,None]
x = torch.randn(8192, 16, device='cuda', dtype=torch.half)
ref = torch.compile(Sequential(
	Linear(x.size(1), x.size(1)*2),
	SiLU(),
	Linear(x.size(1)*2, x.size(1)),
	LogSoftmax(-1),
).to(x.device).requires_grad_(False).eval())
model = torch.compile(Sequential(
	Linear(x.size(1), x.size(1)*4),
	SiLU(),
	Mul(x.size(1)*4, x.size(1)),
	LogSoftmax(-1),
).to(x.device))
scale = torch.GradScaler()
opt = torch.optim.Adam(model.parameters(), .005)
for _ in range(100):
	L = 0
	for _ in range(1000):
		opt.zero_grad()
		x = torch.randn_like(x)
		with torch.autocast(x.device.type, x.dtype):
			t = ref(x)
			y = model(x)
			l = (t-y.transpose(0,1)).flatten(1)@t.exp().ravel()/len(t)
		scale.scale(l.mean()).backward()
		scale.step(opt)
		scale.update()
		L += l.data.float()
	print((L/-1000).exp(), flush=True)
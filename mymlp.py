import torch
from math import gcd
from torch.nn import *
class F(Module):
	def __init__(self, f, *arg):
		super().__init__()
		self.f = f
		self.arg = arg
	def forward(self, x):
		return self.f(x, *self.arg)
class SeqStack(ModuleList):
	def forward(self, x):
		y = [x]
		for self in self:
			x = self(x)
			y.append(x)
		return torch.stack(y)
class Res(Sequential):
	def forward(self, x):
		return super().forward(x)+x
class CausalLinear(Linear):
	def forward(self, x):
		return functional.linear(x, causal(self.weight), self.bias)
def causal(w):
	'''Make w(out*group,in*group) causal: out-group can only see in-group not after it, normalized by std'''
	return(w.reshape(len(w)//(g:=gcd(*w.shape)),g,-1,g).transpose(1,2).tril().transpose(1,2)/torch.arange(1,1+g,dtype=w.dtype,device=w.device).sqrt_()[:,None,None]).reshape(w.shape)
def swiglu(x):
	return (x:=x.chunk(2,-1))[0]*functional.silu(x[1])
torch.manual_seed(42)
x,h,o = torch.empty(16384,16,dtype=torch.half,device='cuda'),16,16
model = [torch.compile(model.to(x.device))for model in[
	Sequential(
		Linear(x.size(1), h),
		SiLU(),
		Linear(h, o),
	).requires_grad_(False).eval(),
	Sequential(
		Linear(x.size(1), h*2),
		SeqStack(Res(
			CausalLinear(h*2, h*2),
			SiLU(),
			CausalLinear(h*2, h*2),
		)for _ in range(2)),
		CausalLinear(h*2, o*h*2)
	),
]]
opt = [torch.optim.Adam(model.parameters(),.005)for model in model[1:]]
scale = [torch.GradScaler()for _ in model[1:]]
for _ in range(100):
	L = [0]*len(opt)
	for _ in range(10000):
		[opt.zero_grad()for opt in opt]
		x = torch.randn_like(x)
		with torch.autocast(x.device.type, x.dtype):
			y = [model(x)for model in model]
			t = y[0].argmax(1)
			l = [functional.cross_entropy(y.unflatten(2,[o,-1]).transpose(1,2),t[:,None].expand(len(y),-1,y.size(2)//o),reduction='none').mean(1)for y in y[1:]]
		[scale.scale(l.mean()).backward()for scale,l in zip(scale,l)]
		[scale.step(opt)for scale,opt in zip(scale,opt)]
		[scale.update()for scale in scale]
		L = [L+l.data for L,l in zip(L,l)]
	print(*[(L/-10000).exp()for L in L], sep='\n')
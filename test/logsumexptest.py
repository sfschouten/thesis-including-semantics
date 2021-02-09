import torch

def logsumexp(x, w):
    c,_ = x.max(dim=2, keepdims=True)
    c,_ = c.max(dim=3, keepdims=True)
    weighted_exp = w * (x - c).exp()
    return c.squeeze() + weighted_exp.sum(dim=2).sum(dim=2).log()


w = torch.ones((1))
def test(x):
    print(f"{torch.logsumexp(x,2)} == {logsumexp(x,w)}")

def x1(x):
    return torch.Tensor([x]).view(1,1,1,1)

test(x1(-1e-15))
test(x1(1e-15))
test(x1(-1))
test(x1(0))
test(x1(1))
test(x1(1e+14))
test(x1(-1e+14))


def x2(*x):
    return torch.Tensor(x).view(1,1,len(x),1)


test(x2(0,0,0,0,0,0))
test(x2(1,1,1,1,1,1))

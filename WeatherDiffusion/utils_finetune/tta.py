# utils/tta.py
import torch

def tta_x8(forward_fn, x):
    rots = [
        lambda t:t,
        lambda t:torch.flip(t,[3]),
        lambda t:torch.flip(t,[2]),
        lambda t:torch.rot90(t,1,[2,3]),
        lambda t:torch.rot90(t,2,[2,3]),
        lambda t:torch.rot90(t,3,[2,3]),
        lambda t:torch.flip(torch.rot90(t,1,[2,3]),[3]),
        lambda t:torch.flip(torch.rot90(t,3,[2,3]),[3]),
    ]
    invs = [
        lambda t:t,
        rots[1],
        rots[2],
        lambda t:torch.rot90(t,3,[2,3]),
        lambda t:torch.rot90(t,2,[2,3]),
        lambda t:torch.rot90(t,1,[2,3]),
        lambda t:rots[1](torch.rot90(t,3,[2,3])),
        lambda t:rots[1](torch.rot90(t,1,[2,3])),
    ]
    outs = []
    with torch.no_grad():
        for r, inv in zip(rots, invs):
            y = forward_fn(r(x))
            outs.append(inv(y))
    return torch.mean(torch.stack(outs, 0), 0)

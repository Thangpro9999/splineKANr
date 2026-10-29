"""Reference values for multi-layer pykan networks (MultKAN), Phase 6.

    python pykan_multilayer_reference.py OUTPUT.json

For each architecture: random coefficients and weights, then forward pass,
loss, gradients of all spline layers, a sample-based grid update of the whole
network, and 5 full-batch Adam steps. Default MultKAN settings (no
multiplication nodes, symbolic branch masked out, node/subnode affine fixed at
scale 1 / bias 0) make MultKAN a plain composition of KANLayers, which is what
splineKANr computes with extrapolate = "none" and bias = FALSE.
"""
import json
import sys

import torch
import kan
from kan.MultKAN import MultKAN

torch.set_default_dtype(torch.float64)
out = {"pykan_version": getattr(kan, "__version__", "unknown"),
       "torch_version": torch.__version__, "cases": []}


def layer_dump(model):
    return [{"grid": l.grid.tolist(), "coef": l.coef.detach().tolist(),
             "scale_base": l.scale_base.detach().tolist(),
             "scale_sp": l.scale_sp.detach().tolist()} for l in model.act_fun]


for seed, (width, grid, k) in enumerate([([2, 3, 1], 5, 3),
                                         ([3, 4, 2, 1], 4, 3),
                                         ([2, 5, 3, 2], 6, 2)]):
    torch.manual_seed(seed)
    model = MultKAN(width=list(width), grid=grid, k=k, seed=seed, auto_save=False,
                    save_act=True, noise_scale=0.5)
    with torch.no_grad():
        for l in model.act_fun:
            l.coef.copy_(torch.randn_like(l.coef) * 0.5)
            l.scale_base.copy_(torch.randn_like(l.scale_base))
            l.scale_sp.copy_(torch.randn_like(l.scale_sp))
    n = 30
    x = torch.rand(n, width[0]) * 2 - 1
    target = torch.randn(n, width[-1])
    init_layers = layer_dump(model)
    y = model(x)
    loss = torch.mean((y - target) ** 2)
    loss.backward()
    grads = [{"coef": l.coef.grad.tolist(),
              "scale_base": l.scale_base.grad.tolist(),
              "scale_sp": l.scale_sp.grad.tolist()} for l in model.act_fun]
    # 5 Adam steps on the spline-layer parameters only
    params = [p for l in model.act_fun
              for p in (l.coef, l.scale_base, l.scale_sp)]
    opt = torch.optim.Adam(params, lr=0.01)
    adam_losses = []
    for step in range(5):
        opt.zero_grad()
        lo = torch.mean((model(x) - target) ** 2)
        lo.backward()
        opt.step()
        adam_losses.append(lo.item())
    after_adam = layer_dump(model)
    # grid update of the whole network on a fresh sample
    xs = torch.rand(200, width[0]) * 1.6 - 0.9
    model.update_grid(xs)
    after_grid = [l.grid.tolist() for l in model.act_fun]
    out["cases"].append({
        "width": width, "grid": grid, "k": k, "x": x.tolist(),
        "target": target.tolist(), "layers": init_layers, "y": y.tolist(),
        "loss": loss.item(), "grads": grads, "adam_lr": 0.01,
        "adam_losses": adam_losses, "after_adam": after_adam,
        "update_x": xs.tolist(), "after_grid": after_grid,
    })

with open(sys.argv[1] if len(sys.argv) > 1 else "pykan_multilayer.json",
          "w") as f:
    json.dump(out, f)

"""Generate numerical reference values from pykan (Phase 6 cross-check).

Run with Python >= 3.9, torch and pykan installed:

    python pykan_reference.py OUTPUT.json

The R test `tests/testthat/test-pykan-fixture.R` reads the JSON and checks
that splineKANr (extrapolate = "none", bias = FALSE, which is the pykan
convention) reproduces pykan's forward pass, coefficient gradients and
sample-based grid update. pykan is only used to produce reference numbers;
splineKANr does not depend on Python.
"""
import json
import sys

import torch
import kan
from kan.KANLayer import KANLayer

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
out = {"pykan_version": getattr(kan, "__version__", "unknown"),
       "torch_version": torch.__version__, "cases": []}

for (in_dim, out_dim, num, k) in [(1, 1, 5, 3), (2, 3, 5, 3), (3, 2, 7, 2)]:
    layer = KANLayer(in_dim=in_dim, out_dim=out_dim, num=num, k=k,
                     noise_scale=1.0)
    with torch.no_grad():
        layer.coef.copy_(torch.randn_like(layer.coef))
        layer.scale_base.copy_(torch.randn_like(layer.scale_base))
        layer.scale_sp.copy_(torch.randn_like(layer.scale_sp))
    x = torch.rand(25, in_dim) * 2.6 - 1.3        # includes extended region
    target = torch.randn(25, out_dim)
    y, preacts, postacts, postspline = layer(x)
    loss = torch.mean((y - target) ** 2)
    loss.backward()
    # sample-based grid update on a fresh copy
    upd = KANLayer(in_dim=in_dim, out_dim=out_dim, num=num, k=k)
    xs = torch.randn(200, in_dim) * 0.4 + 0.1
    upd.update_grid_from_samples(xs)
    out["cases"].append({
        "in_dim": in_dim, "out_dim": out_dim, "num": num, "k": k,
        "grid": layer.grid.tolist(),
        "coef": layer.coef.detach().tolist(),            # (in, out, num + k)
        "scale_base": layer.scale_base.detach().tolist(),  # (in, out)
        "scale_sp": layer.scale_sp.detach().tolist(),
        "x": x.tolist(), "target": target.tolist(),
        "y": y.detach().tolist(),
        "postacts": postacts.detach().tolist(),           # (batch, out, in)
        "loss": loss.item(),
        "grad_coef": layer.coef.grad.tolist(),
        "grad_scale_base": layer.scale_base.grad.tolist(),
        "grad_scale_sp": layer.scale_sp.grad.tolist(),
        "update_x": xs.tolist(),
        "update_grid": upd.grid.tolist(),
    })

with open(sys.argv[1] if len(sys.argv) > 1 else "pykan_reference.json",
          "w") as f:
    json.dump(out, f)

"""Training-level (algorithmic) reference runs of pykan, Phase 6b.

    python pykan_training_reference.py OUT_DIR

Writes two JSON files into OUT_DIR:

* ``pykan_trajectory.json`` -- same initial network, same data, pykan's own
  ``MultKAN.fit(opt="Adam")`` for many steps, with and without pykan's grid
  update schedule. Used to test that ``kan_fit()`` follows the *same
  trajectory* (numerical equivalence of the training algorithm).
* ``pykan_outcomes.json`` -- for several problems and seeds, independent
  pykan fits with pykan's own initialisation (Adam and the default LBFGS)
  and their out-of-sample RMSE, plus the data. Used to compare the
  *distribution* of results with ``kan_fit()`` started from its own
  initialisation (algorithmic equivalence of the procedure, not of the
  numbers).

All computations are float64. Default MultKAN settings (no multiplication
nodes, symbolic branch disabled during fit because lamb = 0, node affine
fixed) make MultKAN a composition of KANLayers, i.e. splineKANr with
extrapolate = "none" and bias = FALSE.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import kan
from kan.MultKAN import MultKAN

torch.set_default_dtype(torch.float64)
out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

PROBLEMS = {
    "sin": (1, lambda x: torch.sin(torch.pi * x[:, [0]])),
    "interaction": (2, lambda x: x[:, [0]] * x[:, [1]]),
    "additive": (2, lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) +
                                          x[:, [1]] ** 2)),
}
WIDTH = {"sin": [1, 1], "interaction": [2, 4, 1], "additive": [2, 5, 1]}


def layer_dump(model):
    return [{"grid": l.grid.tolist(), "coef": l.coef.detach().tolist(),
             "scale_base": l.scale_base.detach().tolist(),
             "scale_sp": l.scale_sp.detach().tolist()} for l in model.act_fun]


def make_data(name, n, gen, noise=0.0):
    p, f = PROBLEMS[name]
    x = torch.rand(n, p, generator=gen) * 2 - 1
    y = f(x) + noise * torch.randn(n, 1, generator=gen)
    return x, y


def mse(model, x, y):
    with torch.no_grad():
        return torch.mean((model(x) - y) ** 2).item()


try:
    from importlib.metadata import version as _v
    PYKAN_VERSION = _v("pykan")
except Exception:
    PYKAN_VERSION = getattr(kan, "__version__", "unknown")
meta = {"pykan_version": PYKAN_VERSION,
        "torch_version": torch.__version__}

# ---- Part A: same init, same data, long trajectories -----------------------
traj_cases = []
for cid, (name, grid, steps, upd) in enumerate([
        ("sin", 5, 300, False), ("interaction", 5, 300, False),
        ("sin", 5, 200, True), ("additive", 5, 200, True)]):
    gen = torch.Generator().manual_seed(100 + cid)
    x, y = make_data(name, 100, gen)
    xt, yt = make_data(name, 100, gen)
    model = MultKAN(width=list(WIDTH[name]), grid=grid, k=3, seed=cid,
                    auto_save=False)
    init = layer_dump(model)
    ds = {"train_input": x, "train_label": y, "test_input": xt,
          "test_label": yt}
    res = model.fit(ds, opt="Adam", lr=0.01, steps=steps, update_grid=upd,
                    log=10 ** 9)
    traj_cases.append({
        "problem": name, "width": WIDTH[name], "grid": grid, "k": 3,
        "steps": steps, "lr": 0.01, "update_grid": upd,
        "grid_update_num": 10, "stop_grid_update_step": 50,
        "x": x.tolist(), "y": y.tolist(), "init": init,
        # pykan logs sqrt(MSE) evaluated *before* each optimiser step
        "rmse_before_step": [float(v) for v in res["train_loss"]],
        "final_mse": mse(model, x, y), "final": layer_dump(model)})
with open(os.path.join(out_dir, "pykan_trajectory.json"), "w") as f:
    json.dump({**meta, "cases": traj_cases}, f)

# ---- Part B: independent fits, distribution of out-of-sample error --------
outcome_cases = []
for name in PROBLEMS:
    for seed in range(1, 11):
        gen = torch.Generator().manual_seed(seed)
        noise = 0.1
        x, y = make_data(name, 500, gen, noise)
        xt, yt = make_data(name, 1000, gen, noise)
        f = PROBLEMS[name][1]
        ds = {"train_input": x, "train_label": y, "test_input": xt,
              "test_label": yt}
        rec = {"problem": name, "seed": seed, "noise_sd": noise,
               "width": WIDTH[name], "x": x.tolist(), "y": y.tolist(),
               "x_test": xt.tolist(), "y_test": yt.tolist(), "fits": {}}
        for opt, steps, lr in [("Adam", 500, 0.01), ("LBFGS", 50, 1.0)]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MultKAN(width=list(WIDTH[name]), grid=5, k=3, seed=seed,
                            auto_save=False)
            t0 = time.time()
            model.fit(ds, opt=opt, lr=lr, steps=steps, log=10 ** 9)
            el = time.time() - t0
            with torch.no_grad():
                pred = model(xt)
            rec["fits"][opt] = {
                "steps": steps, "lr": lr, "seconds": el,
                "test_rmse": torch.sqrt(torch.mean((pred - yt) ** 2)).item(),
                "rmse_vs_truth": torch.sqrt(
                    torch.mean((pred - f(xt)) ** 2)).item()}
        outcome_cases.append(rec)
        print(name, seed, {k: round(v["test_rmse"], 5)
                           for k, v in rec["fits"].items()}, flush=True)
with open(os.path.join(out_dir, "pykan_outcomes.json"), "w") as f:
    json.dump({**meta, "cases": outcome_cases}, f)

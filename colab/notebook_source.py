# %% [markdown]
# # Possession network: {{TITLE}}
#
# This notebook trains one small network on the **open matches** and saves its predictions and its
# weights in the shared Drive folder. It never sees the competition match.
#
# **How to run it**
#
# 1. Menu **Runtime → Change runtime type → T4 GPU → Save**.
# 2. Menu **Runtime → Run all**. Say yes when it asks for access to Google Drive.
# 3. The first run is a short trial. When the last cell prints **TRIAL PASSED**, change
#    `TRIAL = True` to `TRIAL = False` in the first code cell and choose **Run all** again.
# 4. Leave the tab open until the last cell prints **ALL DONE**. If Colab disconnects, choose
#    **Run all** again: rounds that are finished are skipped.
#
# If a cell stops with an error, copy the last lines of the error and send them to Måns.

# %%
MODEL = "{{MODEL}}"     # "tcn" or "bilstm"
TRIAL = True            # True: a short test of every step. False: the full run.
N_SEEDS = 3             # networks trained per round and averaged, as the trees are

FOLDER_NAME = "barcelona_coding"                      # the shared Drive folder
FOLDER_ID = "1fuE9LG_tgV1MhZ8YUPYPuoQ28TlbR8YS"

# %%
# Find the shared folder and a GPU.
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive

drive.mount("/content/drive")


def find_folder():
    def holds_the_pack(place):
        return os.path.exists(os.path.join(place, "network_pack", "meta.json"))

    for place in (f"/content/drive/MyDrive/{FOLDER_NAME}",
                  f"/content/drive/.shortcut-targets-by-id/{FOLDER_ID}/{FOLDER_NAME}",
                  f"/content/drive/.shortcut-targets-by-id/{FOLDER_ID}"):
        if holds_the_pack(place):
            return place
    for pattern in (f"/content/drive/MyDrive/*/{FOLDER_NAME}", f"/content/drive/Shareddrives/*/{FOLDER_NAME}",
                    f"/content/drive/MyDrive/*/*/{FOLDER_NAME}"):
        for place in sorted(glob.glob(pattern)):
            if holds_the_pack(place):
                return place
    raise SystemExit(
        f"Could not find the folder '{FOLDER_NAME}' with 'network_pack' inside it.\n"
        "If the folder is shared with you: open Google Drive, right-click the folder, choose "
        "Organise > Add shortcut > My Drive, then run this cell again.\n"
        "If it is your own folder: check that 'network_pack' has finished uploading into it.")


FOLDER = find_folder()
PACK = os.path.join(FOLDER, "network_pack")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("model:", MODEL, "| trial run" if TRIAL else "| full run")
print("folder:", FOLDER)
print("running on:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU (no GPU)")
if DEVICE == "cpu" and not TRIAL:
    raise SystemExit("No GPU. Choose Runtime > Change runtime type > T4 GPU > Save, then Run all again.")
if DEVICE == "cpu":
    print("No GPU: fine for the trial, only slower. Switch the GPU on before the full run.")

# %%
# Copy the data pack from Drive to this machine's own disk (much faster to read) and check it is complete.
LOCAL = "/content/network_pack"
os.makedirs(LOCAL, exist_ok=True)
with open(os.path.join(PACK, "meta.json")) as handle:
    META = json.load(handle)
started = time.time()
absent = []
for name, info in sorted(META["files"].items()):
    source, target = os.path.join(PACK, name), os.path.join(LOCAL, name)
    if not os.path.exists(source) or os.path.getsize(source) != info["bytes"]:
        absent.append(name)
        continue
    if not os.path.exists(target) or os.path.getsize(target) != info["bytes"]:
        shutil.copyfile(source, target)
needed = [name for name in absent if name.startswith(("train_", "test_"))]
if needed:
    raise SystemExit("These files are missing from network_pack or still uploading: " + ", ".join(needed))
if absent:
    print("Not there yet, so the rounds that need them are skipped for now:", ", ".join(absent))
print(f"{len(META['files']) - len(absent)} files ready in {time.time() - started:.0f}s")

# %%
# {{NETWORK}}

# %%
# Read the pack into memory.
sys.path.insert(0, "/content")
import network  # noqa: E402  (the file written by the cell above)


def read(name):
    with np.load(os.path.join(LOCAL, name), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


OPEN = list(META["open_matches"])
TRAIN = {match: read(f"train_{match}.npz") for match in OPEN}
TEST = {match: read(f"test_{match}.npz") for match in OPEN}
N_INPUTS = len(META["names"])
COPIES = int(META["copies_per_match"])
SWAP_SOURCE = torch.tensor(META["swap"]["source"], dtype=torch.long, device=DEVICE)
SWAP_SIGN = torch.tensor(META["swap"]["sign"], dtype=torch.float32, device=DEVICE)
SWAP_OFFSET = torch.tensor(META["swap"]["offset"], dtype=torch.float32, device=DEVICE)
print(f"{len(OPEN)} open matches, {COPIES} damaged copies each, {N_INPUTS} inputs, "
      f"{sum(len(TRAIN[m]['y']) for m in OPEN):,} moments a copy")

# %%
# The two networks and their settings. Both look only a few seconds to either side of a moment.
SETTINGS = {
    "tcn": dict(kind="tcn", channels=64, dilations=[1, 2, 4, 8], dropout=0.2, length=256, margin=30, batch=64),
    "bilstm": dict(kind="bilstm", proj=96, hidden=64, layers=1, window=64, hop=32, dropout=0.2,
                   length=64, margin=16, batch=256),
}
CFG = SETTINGS[MODEL]
CLIP = 8.0                 # scaled inputs are cut off at this many standard deviations
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP = 200               # steps over which the learning rate rises to its value
AVERAGING = 0.999          # the saved weights are a running average of the last ~1000 steps
MAX_STEPS = 300 if TRIAL else 8000
CHECK_EVERY = 100 if TRIAL else 250
PATIENCE = 2 if TRIAL else 6
VALIDATION_COPIES = (0, 3, 6)
SEEDS = 1 if TRIAL else N_SEEDS


class Block(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        y = self.drop(F.relu(self.conv1(h)))
        y = self.drop(F.relu(self.conv2(y)))
        return h + y


class TCN(nn.Module):
    def __init__(self, n_in, channels, dilations, dropout):
        super().__init__()
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList([Block(channels, d, dropout) for d in dilations])
        self.out = nn.Conv1d(channels, 3, 1)

    def forward(self, z):                           # [batch, steps, inputs]
        h = F.relu(self.inp(z.transpose(1, 2)))
        for block in self.blocks:
            h = block(h)
        return self.out(h).transpose(1, 2)          # [batch, steps, 3]


class BiLSTM(nn.Module):
    def __init__(self, n_in, proj, hidden, layers, dropout):
        super().__init__()
        self.inp = nn.Linear(n_in, proj)
        self.lstm = nn.LSTM(proj, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 3)
        self.drop = nn.Dropout(dropout)

    def forward(self, z):                           # [batch, steps, inputs]
        h = self.drop(F.relu(self.inp(z)))
        h, _ = self.lstm(h)
        return self.out(self.drop(h))               # [batch, steps, 3]


def make_model():
    n_in = 2 * N_INPUTS                             # every input, and a flag saying whether it is missing
    if CFG["kind"] == "tcn":
        return TCN(n_in, CFG["channels"], CFG["dilations"], CFG["dropout"]).to(DEVICE)
    return BiLSTM(n_in, CFG["proj"], CFG["hidden"], CFG["layers"], CFG["dropout"]).to(DEVICE)


print(MODEL, "with", f"{sum(p.numel() for p in make_model().parameters()):,}", "weights")

# %%
# Data handling: training windows, scaling, swapping the teams.
def half_bounds(half):
    """(first row, one past the last row) of each half."""
    edges = [0] + (np.flatnonzero(np.diff(half)) + 1).tolist() + [len(half)]
    return list(zip(edges[:-1], edges[1:]))


def training_set(matches):
    """Every damaged copy of these matches on the GPU, and where each half of each copy starts."""
    total = sum(len(TRAIN[m]["y"]) for m in matches) * COPIES
    X = torch.empty((total, N_INPUTS), dtype=torch.float32, device=DEVICE)
    Y = torch.empty(total, dtype=torch.long, device=DEVICE)
    W = torch.empty(total, dtype=torch.float32, device=DEVICE)
    starts, lengths, at = [], [], 0
    for match in matches:
        data = TRAIN[match]
        n = len(data["y"])
        for copy in range(COPIES):
            X[at:at + n] = torch.from_numpy(data[f"x{copy}"]).to(DEVICE)
            Y[at:at + n] = torch.from_numpy(data["y"].astype(np.int64)).to(DEVICE)
            W[at:at + n] = torch.from_numpy(data["w"]).to(DEVICE)
            for low, high in half_bounds(data["half"]):
                if high - low >= CFG["length"]:
                    starts.append(at + low)
                    lengths.append(high - low)
            at += n
    lengths = torch.tensor(lengths, dtype=torch.long, device=DEVICE)
    return {"X": X, "Y": Y, "W": W, "start": torch.tensor(starts, dtype=torch.long, device=DEVICE),
            "length": lengths, "chance": lengths.float() / lengths.float().sum()}


def column_stats(X, chunk=200_000):
    """Mean and spread of every input over the values that are present."""
    total = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    squares = torch.zeros_like(total)
    count = torch.zeros_like(total)
    for at in range(0, X.shape[0], chunk):
        part = X[at:at + chunk].double()
        present = torch.isfinite(part)
        part = torch.where(present, part, torch.zeros_like(part))
        total += part.sum(0)
        squares += (part * part).sum(0)
        count += present.sum(0)
    count = count.clamp(min=1.0)
    mean = total / count
    std = (squares / count - mean * mean).clamp(min=0.0).sqrt().clamp(min=1e-3)
    return mean.float(), std.float()


def transform(x, mean, std):
    """Scaled inputs with 0 where one is missing, followed by the missing flags."""
    missing = torch.isnan(x)
    z = ((x - mean) / std).clamp(-CLIP, CLIP)
    return torch.cat([torch.where(missing, torch.zeros_like(z), z), missing.to(z.dtype)], dim=-1)


def swap_inputs(x):
    return x[..., SWAP_SOURCE] * SWAP_SIGN + SWAP_OFFSET


def swap_labels(y):
    return torch.where(y == 0, torch.ones_like(y), torch.where(y == 1, torch.zeros_like(y), y))


def batch_of(S, generator):
    """Random windows from random halves, half of them with home and away exchanged."""
    size, length = CFG["batch"], CFG["length"]
    segment = torch.multinomial(S["chance"], size, replacement=True, generator=generator)
    room = S["length"][segment] - length + 1
    offset = (torch.rand(size, device=DEVICE, generator=generator) * room.float()).long()
    start = S["start"][segment] + torch.minimum(offset, room - 1)
    index = start[:, None] + torch.arange(length, device=DEVICE)[None, :]
    x, y, w = S["X"][index], S["Y"][index], S["W"][index]
    flip = torch.rand(size, device=DEVICE, generator=generator) < 0.5
    x = torch.where(flip[:, None, None], swap_inputs(x), x)
    y = torch.where(flip[:, None], swap_labels(y), y)
    return x, y, w


def loss_of(logits, y, w):
    """Weighted like the trees' training. The ends of a window are left out: they lack context on one side."""
    margin = CFG["margin"]
    stop = logits.shape[1] - margin
    logits, y, w = logits[:, margin:stop], y[:, margin:stop], w[:, margin:stop]
    each = F.cross_entropy(logits.reshape(-1, 3), y.clamp(max=2).reshape(-1), reduction="none")
    w = w.reshape(-1)
    return (each * w).sum() / w.sum().clamp(min=1e-6)

# %%
# Prediction, and the scores printed along the way.
@torch.no_grad()
def logits_for(model, x, half, mean, std):
    """The network's raw answer [n, 3] for inputs as given (numpy in, numpy out). Each half on its own."""
    model.eval()
    out = np.zeros((len(x), 3), dtype=np.float32)
    for low, high in half_bounds(half):
        z = transform(torch.from_numpy(np.ascontiguousarray(x[low:high], dtype=np.float32)).to(DEVICE), mean, std)
        if CFG["kind"] == "tcn":
            out[low:high] = model(z[None])[0].float().cpu().numpy()
        else:
            starts, which = network.window_plan(high - low, CFG["window"], CFG["hop"])
            steps = min(high - low, CFG["window"])
            index = torch.from_numpy(starts).to(DEVICE)[:, None] + torch.arange(steps, device=DEVICE)[None, :]
            parts = [model(z[index[at:at + 512]]) for at in range(0, index.shape[0], 512)]
            out[low:high] = network.read_windows(torch.cat(parts).float().cpu().numpy(), starts, which)
    return out


def softmax(logits, temperature=1.0):
    scaled = logits.astype(np.float64) / temperature
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def probabilities_for(models, x, half, mean, std, temperature):
    """H, A, D at each moment: averaged over the networks, and over the match as given and with teams swapped."""
    swapped = network.swap(x.astype(np.float64), META["swap"]).astype(np.float32)
    total = np.zeros((len(x), 3))
    for model in models:
        total += softmax(logits_for(model, x, half, mean, std), temperature)
        total += softmax(logits_for(model, swapped, half, mean, std), temperature)[:, [1, 0, 2]]
    return (total / (2 * len(models))).astype(np.float32)


def log_loss(prob, y, w):
    keep = w > 0
    return float(-(w[keep] * np.log(prob[keep, y[keep]] + 1e-12)).sum() / w[keep].sum())


def accuracy(prob, y, w):
    keep = w > 0
    return float((prob[keep].argmax(1) == y[keep]).mean())


def validation_loss(model, match, mean, std, temperature=1.0):
    data = TRAIN[match]
    return float(np.mean([log_loss(softmax(logits_for(model, data[f"x{c}"], data["half"], mean, std), temperature),
                                   data["y"], data["w"]) for c in VALIDATION_COPIES]))


def best_temperature(model, match, mean, std):
    """How much to soften the answers so that they are as sure as they are right, measured on the validation match."""
    data = TRAIN[match]
    logits = [logits_for(model, data[f"x{c}"], data["half"], mean, std) for c in VALIDATION_COPIES]
    grid = [round(0.3 + 0.05 * k, 2) for k in range(55)]          # 0.30 to 3.00
    losses = [np.mean([log_loss(softmax(l, t), data["y"], data["w"]) for l in logits]) for t in grid]
    return float(grid[int(np.argmin(losses))])

# %%
# Training one network.
def train(matches, steps, seed, validation=None, label=""):
    """Train on these matches. With a validation match: stop when it stops improving and report the best step.

    Returns the averaged weights, the step they are from, the input scaling and the validation loss.
    """
    S = training_set(matches)
    mean, std = column_stats(S["X"])
    torch.manual_seed(seed)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    model = make_model()
    average = make_model()
    average.load_state_dict(model.state_dict())
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_loss, best_step, best_state, waited, running, began = float("inf"), 0, None, 0, 0.0, time.time()
    for step in range(1, steps + 1):
        model.train()
        for group in optimiser.param_groups:
            group["lr"] = LEARNING_RATE * min(1.0, step / WARMUP)
        x, y, w = batch_of(S, generator)
        loss = loss_of(model(transform(x, mean, std)), y, w)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        keep = min(AVERAGING, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for slow, fast in zip(average.parameters(), model.parameters()):
                slow.mul_(keep).add_(fast.detach(), alpha=1.0 - keep)
        running += float(loss.item())
        if step % CHECK_EVERY == 0:
            line = f"  {label} step {step:>5}  training loss {running / CHECK_EVERY:.4f}"
            running = 0.0
            if validation is not None:
                now = validation_loss(average, validation, mean, std)
                if now < best_loss - 1e-4:
                    best_loss, best_step, waited = now, step, 0
                    best_state = {k: v.detach().clone() for k, v in average.state_dict().items()}
                else:
                    waited += 1
                line += f"  validation {now:.4f}  best {best_loss:.4f} at step {best_step}"
            print(line + f"  ({time.time() - began:.0f}s)", flush=True)
            if validation is not None and waited >= PATIENCE:
                break
    del S
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if validation is None:
        return {k: v.detach().clone() for k, v in average.state_dict().items()}, steps, mean, std, None
    if best_state is None:
        raise RuntimeError("the validation loss never improved: something is wrong with the training")
    return best_state, best_step, mean, std, best_loss


def with_weights(state):
    model = make_model()
    model.load_state_dict(state)
    return model.eval()

# %%
# One round: choose the length of training on a validation match, train the networks on every match of the
# round for that long, predict the round's test matches, save.
RESULTS = os.path.join(FOLDER, "results", MODEL + ("_trial" if TRIAL else ""))
os.makedirs(RESULTS, exist_ok=True)


def model_config(temperature, seeds, matches, steps):
    config = {k: v for k, v in CFG.items() if k not in ("dropout", "length", "margin", "batch")}
    config.update(names=META["names"], swap={k: META["swap"][k] for k in ("source", "sign", "offset")},
                  mask_columns=list(range(N_INPUTS)), clip=CLIP, temperature=temperature, seeds=seeds,
                  trained_on=list(matches), steps=int(steps), torch=torch.__version__)
    return config


def save_model(path, states, mean, std, temperature, matches, steps):
    network.save(path, model_config(temperature, len(states), matches, steps), mean.cpu().numpy(), std.cpu().numpy(),
                 [{k: v.detach().cpu().numpy() for k, v in state.items()} for state in states])


def run_round(name, matches, targets):
    folder = os.path.join(RESULTS, name)
    done = os.path.join(folder, "info.json")
    if os.path.exists(done):
        print(f"{name}: already done, skipped")
        return None
    os.makedirs(folder, exist_ok=True)
    began = time.time()
    matches = list(matches)
    validation = sorted(matches)[sum(ord(c) for c in name) % len(matches)]
    rest = [m for m in matches if m != validation]
    print(f"\n{name}: {len(matches)} training matches; choosing the length of training with {validation} held back")
    state, best_step, mean, std, best_loss = train(rest, MAX_STEPS, 1000, validation, "search")
    temperature = best_temperature(with_weights(state), validation, mean, std)
    steps = int(round(best_step * len(matches) / len(rest)))
    print(f"{name}: best at step {best_step}, temperature {temperature}; now {SEEDS} network(s) on all "
          f"{len(matches)} matches for {steps} steps")
    states = []
    for seed in range(SEEDS):
        state, _, mean, std, _ = train(matches, steps, seed, None, f"network {seed + 1}/{SEEDS}")
        states.append(state)
    models = [with_weights(state) for state in states]

    scores = {}
    for kind, match in targets:
        if kind == "stress":
            data = read(f"stress_{match}.npz")
            out = {}
            for index, condition in enumerate(META["stress_conditions"]):
                out[f"prob{index}"] = probabilities_for(models, data[f"x{index}"], data["half"], mean, std, temperature)
                scores[f"{match}, {condition}"] = {"log_loss": log_loss(out[f"prob{index}"], data["y"], data["w"]),
                                                   "accuracy": accuracy(out[f"prob{index}"], data["y"], data["w"])}
            np.savez_compressed(os.path.join(folder, f"stress_{match}.npz"), **out)
            continue
        data = TEST[match] if kind == "test" else read(f"sc_{match}.npz")
        prob = probabilities_for(models, data["x"], data["half"], mean, std, temperature)
        np.savez_compressed(os.path.join(folder, f"{kind}_{match}.npz"), prob=prob)
        scores[match] = {"log_loss": log_loss(prob, data["y"], data["w"]), "accuracy": accuracy(prob, data["y"], data["w"])}
        print(f"  {match}: log loss {scores[match]['log_loss']:.4f}, right at {100 * scores[match]['accuracy']:.1f}% of moments")
    save_model(os.path.join(folder, f"network_model_{MODEL}.npz"), states, mean, std, temperature, matches, steps)
    info = {"round": name, "model": MODEL, "trained_on": matches, "validation_match": validation,
            "best_step": int(best_step), "steps": int(steps), "temperature": temperature,
            "validation_loss": best_loss, "seeds": SEEDS, "scores": scores,
            "minutes": round((time.time() - began) / 60, 1)}
    with open(done, "w") as handle:
        json.dump(info, handle, indent=1)
    print(f"{name}: done in {info['minutes']} minutes")
    return models, states, mean, std, temperature


def check_numpy(folder):
    """Does the plain-numpy prediction code give the same answers as the trained network? Saves a reference."""
    saved = network.load(os.path.join(folder, f"network_model_{MODEL}.npz"))
    weights = [{k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in seed.items()} for seed in saved["seeds"]]
    models = [with_weights(state) for state in weights]
    mean = torch.tensor(saved["mean"], dtype=torch.float32, device=DEVICE)
    std = torch.tensor(saved["std"], dtype=torch.float32, device=DEVICE)
    data = TEST[OPEN[0]]
    rows = np.concatenate([np.arange(low, min(high, low + 1500)) for low, high in half_bounds(data["half"])])
    x, half = data["x"][rows], data["half"][rows]
    by_torch = probabilities_for(models, x, half, mean, std, saved["config"]["temperature"])
    by_numpy = network.probabilities(saved, x, half, decimals=None)
    gap = float(np.abs(by_torch - by_numpy).max())
    np.savez_compressed(os.path.join(folder, f"network_reference_{MODEL}.npz"), x=x, half=half, prob=by_torch)
    print(f"numpy check: largest difference {gap:.1e}  ->  {'PASSED' if gap < 1e-4 else 'NOT THE SAME, tell Måns'}")
    return gap < 1e-4

# %%
# The rounds. Each is saved to Drive as soon as it is finished.
DFL = [m for m in META["dfl"] if m in OPEN]
METRICA = [m for m in META["metrica"] if m in OPEN]
STRESS = list(META.get("stress_held_out", []))
ROUNDS = [(f"fold_{match}", [m for m in OPEN if m != match], [("test", match)]) for match in OPEN]
ROUNDS.append(("all", OPEN, [("sc", match) for match in META.get("skillcorner", [])]))
ROUNDS.append(("stress", [m for m in OPEN if m not in STRESS], [("stress", match) for match in STRESS]))
ROUNDS.append(("dfl_only", DFL, [("test", match) for match in METRICA]))
ROUNDS.append(("metrica_only", METRICA, [("test", match) for match in DFL]))
if TRIAL:
    ROUNDS = ROUNDS[:1]

everything = time.time()
skipped = []
for name, matches, targets in ROUNDS:
    files = [f"{kind}_{match}.npz" for kind, match in targets if kind != "test"]
    if any(f in absent for f in files) or (name == "stress" and not STRESS):
        skipped.append(name)
        print(f"\n{name}: skipped, its files are not in network_pack yet")
        continue
    run_round(name, matches, targets)

passed = check_numpy(os.path.join(RESULTS, ROUNDS[0][0] if TRIAL else "all")) \
    if os.path.exists(os.path.join(RESULTS, ROUNDS[0][0] if TRIAL else "all", "info.json")) else False

print(f"\ntotal time {(time.time() - everything) / 60:.0f} minutes; results are in {RESULTS}")
if TRIAL:
    print("TRIAL PASSED: now set TRIAL = False in the first code cell and choose Runtime > Run all."
          if passed else "TRIAL NOT PASSED: copy the lines above and send them to Måns.")
elif skipped or not passed:
    print("NOT FINISHED:", ("skipped " + ", ".join(skipped)) if skipped else "the numpy check did not pass", "- tell Måns.")
else:
    print("ALL DONE. Tell Måns the run is finished.")

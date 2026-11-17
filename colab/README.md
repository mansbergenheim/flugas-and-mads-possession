# Training the two networks on Google Colab

Two small networks are trained on the **open matches only**: a TCN and a BiLSTM. One person runs each.
Nothing here touches the competition match. It is predicted later, on our own computers, from the
weights these notebooks save.

| File | What it is |
|---|---|
| `train_tcn.ipynb` | The notebook for the TCN |
| `train_bilstm.ipynb` | The notebook for the BiLSTM. The same notebook with one line changed |
| `notebook_source.py`, `make_notebooks.py` | What the two notebooks are generated from |
| `../scripts/export_network_pack.py` | Writes the data pack the notebooks read (`runs/network_pack`) |
| `../mtah/network.py` | The prediction code for our own computers, in plain numpy. Each notebook carries a copy and checks that it reproduces the trained network |

## Once, by the person who has the project on their computer

1. Open the shared Drive folder `barcelona_coding` in a browser.
2. Drag the folder `possession/runs/network_pack` from Finder into it. It is 34 files, about 1.9 GB.
   Wait until Drive says the upload is complete.
3. Check that `train_tcn.ipynb` and `train_bilstm.ipynb` are in `barcelona_coding`. If not, drag them in
   from `possession/colab`.

## Each of you

1. **Only if the folder is not your own:** in Drive, right-click `barcelona_coding`, choose
   **Organise → Add shortcut → My Drive**. Colab can only see what is in "My Drive".
2. Double-click your notebook and choose **Open with → Google Colaboratory**. The first time you may
   have to pick **Connect more apps** and add Colaboratory.
3. Menu **Runtime → Change runtime type → T4 GPU → Save**.
4. Menu **Runtime → Run all**. When it asks for access to Google Drive, choose your account and allow it.
5. The first run is a trial of a few minutes. It ends with **TRIAL PASSED**.
6. In the first code cell change `TRIAL = True` to `TRIAL = False`, then **Runtime → Run all** again.
7. Leave the tab open. The run ends with **ALL DONE**.

If Colab disconnects or a cell stops, choose **Run all** again. Rounds that are finished are skipped.
If it stops with an error, copy the last lines and send them on. They can be pasted anywhere: only
open data is involved.

## Afterwards

In Drive, right-click `barcelona_coding/results`, choose **Download**, and unzip it into
`possession/runs/network_results/`. Each network has its own folder there, with one folder per round.

Then, from the `possession` folder, bring each network's predictions into the scoring format and
compare everything. Open data only, so these can be run by anyone:

```bash
python3 scripts/import_network_results.py --model tcn
```

```bash
python3 scripts/import_network_results.py --model bilstm
```

```bash
python3 experiments/compare_networks.py
```

The last one prints the trees, each network, and every combination on the five tests, and says which
formats pass the agreed rule. The rule ranks kinds of format (a plain average, then weighted, then
fused). If several of the simplest kind pass, it lists them all, and choosing between them is the
team's decision.

A follow-up for understanding the result, not for picking a weight:

```bash
python3 experiments/compare_networks_detail.py
```

It prints how often the trees and the networks give a different answer, the plain averages head to
head, and what happens as the trees' share goes from nothing to everything.

## What a notebook does

It trains its network 14 times, each time on different matches, so that every score we compare is
on matches the network has not seen:

| Rounds | Trained on | Predicts |
|---|---|---|
| 10 × `fold_<match>` | The other 9 open matches | That match's standard test copy |
| `all` | All 10 open matches | SkillCorner's working half. These are the weights we would ship |
| `stress` | The 7 matches outside the stress table | The 3 stress matches under all 19 kinds of damage |
| `dfl_only`, `metrica_only` | One source | The other source's matches |

In each round it first holds one training match back to find out how long to train, then trains three
networks on all of the round's matches for that long and averages them.

## Rules

- Only open matches. Never put competition files in this Drive folder, and never open them in Colab.
- Change nothing in the notebook except `TRIAL`.

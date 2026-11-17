# The trained network the submission uses

`tcn.npz` holds the TCN half of the submitted model: three small networks, trained from different
random starts and averaged at prediction time. `mtah/candidate.py` loads it and refuses any file
whose SHA-256 is not the pinned one.

| | |
|---|---|
| Made by | `colab/train_tcn.ipynb`, round `all`, on Google Colab's free T4 GPU, 11 November 2026 |
| Trained on | The ten open matches (seven DFL, three Metrica), on the same damaged copies and the same 198 inputs as the tree models. No competition data |
| Inside | Three networks of about 124,000 weights each, the input scaling, and the settings needed to run them |
| Training length | 5,833 steps, found with the match J03WOY held back, then used for all three networks on all ten matches |
| Temperature | 1.1, fitted on the held-back match |
| SHA-256 | `b451ff1d19ec15ba1b8692912636474b91cf2371631516d59d39c88ceb374478` |

## Why it ships as a file

Training a network on a GPU does not repeat to the last bit, so running the notebook again gives a
network that is as good and not identical. The submission has to be rebuilt exactly, so the weights
are part of the repository, and the notebook that made them sits in `colab/`.

Prediction from the file needs no GPU and no PyTorch. `mtah/network.py` does it in plain numpy and
rounds the probabilities to four decimals, so every machine gets the same numbers.
`tests/test_network.py` checks that the numpy code reproduces what PyTorch answered on Colab, from
the reference files in `tests/fixtures/`.

## Making it again

Follow `colab/README.md`. The data pack the notebook reads is written by
`scripts/export_network_pack.py` from the open matches. A fresh run would need its new file copied
here and its SHA-256 put in `mtah/candidate.py`, and every test of the model would have to be run
again on it.

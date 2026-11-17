"""Write the two training notebooks from one source, so that they differ in one line only.

colab/notebook_source.py holds the cells, separated by lines that start with "# %%". The prediction
code the submission uses (mtah/network.py) is copied into each notebook as a cell, so the notebook can
check on the spot that it reproduces the network it has just trained.

    python3 colab/make_notebooks.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TITLES = {"tcn": "TCN", "bilstm": "BiLSTM"}
NETWORK_CELL = "# {{NETWORK}}"


def cells(source, network):
    out = []
    for chunk in ("\n" + source).split("\n# %%")[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.split("\n"))
            out.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
            continue
        if body == NETWORK_CELL:
            body = "%%writefile /content/network.py\n" + network.rstrip("\n")
        else:
            compile(body, "<notebook cell>", "exec")        # a syntax error should stop here, not on Colab
        out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                    "source": body.splitlines(keepends=True)})
    return out


if __name__ == "__main__":
    with open(os.path.join(HERE, "notebook_source.py")) as handle:
        source = handle.read()
    with open(os.path.join(ROOT, "mtah", "network.py")) as handle:
        network = handle.read()
    compile(network, "network.py", "exec")
    for model, title in TITLES.items():
        text = source.replace("{{MODEL}}", model).replace("{{TITLE}}", title)
        notebook = {"cells": cells(text, network), "nbformat": 4, "nbformat_minor": 0,
                    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                 "language_info": {"name": "python"}}}
        path = os.path.join(HERE, f"train_{model}.ipynb")
        with open(path, "w") as handle:
            json.dump(notebook, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        print(f"{path}: {len(notebook['cells'])} cells")

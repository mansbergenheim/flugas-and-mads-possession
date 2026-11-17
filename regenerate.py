"""Regenerate the submitted CSV from the match files, on any computer, with one command.

    Windows:      py -3.14 regenerate.py --data C:\\path\\to\\match-data
    Mac / Linux:  python3.14 regenerate.py --data /path/to/match-data

The match-data folder is the organisers' bundle laid out as tactical/ (BAR-ATH_tactical_atd.txt,
BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_smart_tagging.xml). It must be outside this repository.
What is built, and the hash it must come out with, is in submission.json.

In order, printing as it goes:
  1. checks the Python version (3.14; the file was built with 3.14.3)
  2. makes a private environment in ./.venv with the exact pinned packages (requirements.txt), plus
     pandas for the organisers' validator (requirements-validator.txt)
  3. downloads the open training data (DFL and Metrica, about 2.6 GB) and checks every file's SHA-256
  4. runs the unit tests (for information)
  5. builds the model from open data and compares its five fingerprints with the build Mac's
     (skip with --skip-fingerprints; the build in step 6 does the same training again)
  6. builds the submission: scripts/make_submission.py with the settings in submission.json, the
     committed commentary cues included if submission.json names them
  7. runs the organisers' own validator (downloaded at its pinned commit, hash-checked) on the file
  8. compares the file's SHA-256 with the one in submission.json and writes regenerate_result.txt

About 2.6 GB of download, then roughly an hour of computing. 8 GB of memory is enough. Steps 1-5 use
public data only. The CSV it writes is about the competition match: don't share it.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(REPO, ".venv")
OPEN_DIR = os.environ.get("MTAH_OPEN_DIR", os.path.join(REPO, "data_open"))
VALIDATOR_DIR = os.path.join(REPO, ".validator")
RESULT = os.path.join(REPO, "regenerate_result.txt")

# The build Mac's fingerprints of the model built from open data (README, "Checking another computer").
FINGERPRINTS = {"test copy": "8a1175fa39ad97cc", "trees": "dc8b1af3e77046ae", "tcn": "f602821ad3381bd7",
                "probabilities": "b4048da20b2b3da1", "labels": "5bb1373d7775153e"}

# The same pinned sources as scripts/download_open_data.sh.
METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/e706dd506b360d69d9d123d5b8026e7294b13996"
METRICA_FILES = ["documentation/events-definitions.pdf",
                 "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
                 "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
                 "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
                 "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
                 "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
                 "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
                 "data/Sample_Game_3/Sample_Game_3_events.json",
                 "data/Sample_Game_3/Sample_Game_3_metadata.xml",
                 "data/Sample_Game_3/Sample_Game_3_tracking.txt"]
DFL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/a715a38dfbaf5f58e431727c2b78d174101a703c"
DFL_MATCHES = ["000002_DFL-MAT-J03WPY", "000002_DFL-MAT-J03WOH", "000002_DFL-MAT-J03WOY",
               "000002_DFL-MAT-J03WQQ", "000002_DFL-MAT-J03WR9", "000001_DFL-MAT-J03WMX",
               "000001_DFL-MAT-J03WN1"]
DFL_KINDS = ["02_01_matchinformation", "03_02_events_raw", "04_03_positions_raw_observed"]

# The organisers' validator at the latest commit of their repository (23 Oct 2026), hash-checked.
OFFICIAL = "https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/f8f2629ff1e0985f81d7e59d8413678514ee1530"
VALIDATOR = {"games.py": "3b2122794e9e0476e2b3a871d14880e5fa47c4feaeacf99caef054c2309076ce",
             "evaluation/possession_metrics.py": "341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d",
             "evaluation/validate_submission.py": "07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14"}

log = []


def say(text=""):
    print(text, flush=True)
    log.append(text)


def step(text):
    say(f"\n=== {text}")


def stop(text):
    say(f"\nSTOPPED: {text}")
    write_result()
    sys.exit(1)


def write_result():
    with open(RESULT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(log) + "\n")


def venv_python():
    return os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target):
    part = target + ".part"
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mtah-regenerate"})
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
            os.replace(part, target)
            return
        except OSError as error:
            print(f"    attempt {attempt} failed: {type(error).__name__}", flush=True)
            time.sleep(5 * attempt)
    stop(f"could not download {os.path.basename(target)}. Check the internet connection and run this again.")


def fetch_checked(pairs, label):
    """Download (url, target, sha256) triples, skipping files already there and correct."""
    for i, (url, target, want) in enumerate(pairs, 1):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        name = os.path.relpath(target, REPO)
        if os.path.exists(target) and sha256(target) == want:
            print(f"  [{i:2d}/{len(pairs)}] already there: {name}", flush=True)
            continue
        print(f"  [{i:2d}/{len(pairs)}] downloading {name}", flush=True)
        download(url, target)
        if sha256(target) != want:
            stop(f"{name} downloaded but its SHA-256 is wrong. Delete it and run this again.")
    say(f"all {len(pairs)} {label} match their pinned SHA-256")


def run(args, **kw):
    """Run a command in the private environment, showing its output; return (exit code, lines)."""
    # UTF-8 both ways, so a Windows console's code page can't stop a run over one printed character.
    process = subprocess.Popen([venv_python(), *args], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace",
                               env=dict(os.environ, MTAH_OPEN_DIR=OPEN_DIR, PYTHONUTF8="1", **kw))
    lines = []
    for line in process.stdout:
        print("  " + line.rstrip(), flush=True)
        lines.append(line.rstrip())
    return process.wait(), lines


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="the match-data folder holding tactical/")
    ap.add_argument("--out", default=None, help="where to write the CSV (default: <data>/regenerated)")
    ap.add_argument("--skip-fingerprints", action="store_true", help="skip step 5")
    args = ap.parse_args()
    data = os.path.abspath(args.data)
    out = os.path.abspath(args.out or os.path.join(data, "regenerated"))
    with open(os.path.join(REPO, "submission.json"), encoding="utf-8") as handle:
        plan = json.load(handle)
    say(f"Regenerate, {time.strftime('%Y-%m-%d %H:%M')}: {plan['team']}.csv")

    step("1/8 Python")
    say(f"Python {platform.python_version()} on {platform.platform()} ({platform.machine()})")
    if sys.version_info[:2] != (3, 14):
        stop("this needs Python 3.14 (the file was built with 3.14.3): https://www.python.org/downloads/release/"
             "python-3143/ then run again with py -3.14 (Windows) or python3.14 (Mac/Linux).")
    if platform.python_version() != "3.14.3":
        say("Note: the build used 3.14.3 exactly. Carrying on; another 3.14 release very rarely matters.")
    for name in ("tactical/BAR-ATH_tactical_atd.txt", "tactical/BAR-ATH_tactical_atd.xml",
                 "tactical/BAR-ATH_tactical_smart_tagging.xml"):
        if not os.path.exists(os.path.join(data, *name.split("/"))):
            stop(f"{name} not found in {data}")

    step("2/8 Private environment with the pinned packages")
    if not os.path.exists(venv_python()):
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    for req in ("requirements.txt", "requirements-validator.txt"):
        if subprocess.run([venv_python(), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r",
                           os.path.join(REPO, req)]).returncode:
            stop(f"installing {req} failed (see the lines above).")
    versions = subprocess.run([venv_python(), "-c", "import numpy, scipy, sklearn, pandas; print(numpy.__version__, "
                               "scipy.__version__, sklearn.__version__, pandas.__version__)"],
                              capture_output=True, text=True, check=True).stdout.strip()
    say(f"numpy, scipy, scikit-learn, pandas: {versions}")

    step("3/8 Open training data (about 2.6 GB the first time)")
    with open(os.path.join(REPO, "mtah", "open_data_sha256.json")) as handle:
        expected = json.load(handle)
    wanted = [(f"{METRICA}/{f}", "metrica/" + os.path.basename(f)) for f in METRICA_FILES]
    wanted += [(f"{DFL}/DFL_{k}_DFL-COM-{m}.xml", f"dfl/DFL_{k}_DFL-COM-{m}.xml") for m in DFL_MATCHES for k in DFL_KINDS]
    if sorted(name for _, name in wanted) != sorted(expected):
        stop("the download list doesn't match mtah/open_data_sha256.json.")
    fetch_checked([(url, os.path.join(OPEN_DIR, *name.split("/")), expected[name]) for url, name in wanted],
                  "open files")

    step("4/8 Unit tests (information only)")
    code, lines = run(["-m", "unittest", "discover", "-s", "tests", "-q"])
    tests = "OK" if code == 0 else "FAILED: " + " / ".join(lines[-3:])
    say(f"unit tests: {tests}")

    step("5/8 Model fingerprints from open data")
    if args.skip_fingerprints:
        say("skipped (--skip-fingerprints)")
        same_model = None
    else:
        code, lines = run([os.path.join("scripts", "repro_check.py")])
        if code:
            stop("the fingerprint build failed (see the lines above).")
        found = {k: next((l.split()[-1] for l in lines if l.startswith(k + " ")), "missing") for k in FINGERPRINTS}
        for k, want in FINGERPRINTS.items():
            say(f"  {k:<14}{found[k]:<20}Mac {want:<20}{'same' if found[k] == want else 'DIFFERENT'}")
        same_model = found == FINGERPRINTS
        if not same_model:
            first = next(k for k in FINGERPRINTS if found[k] != FINGERPRINTS[k])
            say(f"the first difference is '{first}' (see the README on what each line covers); the build carries "
                "on, and step 8 says whether the file itself still comes out the same")

    step("6/8 Building the submission (about 30 minutes)")
    command = [os.path.join("scripts", "make_submission.py"), "--team", plan["team"], "--out", out]
    if plan.get("cues"):
        command += ["--cues", os.path.join(REPO, plan["cues"]),
                    "--team-weight", str(plan["team_weight"]), "--dead-weight", str(plan["dead_weight"])]
    say("make_submission.py " + " ".join(command[1:]).replace(data, "<data>").replace(REPO + os.sep, ""))
    code, lines = run(command, MTAH_DATA_DIR=data)
    if code:
        stop("the build failed (see the lines above).")
    target = os.path.join(out, plan["team"] + ".csv")

    step("7/8 The organisers' validator")
    fetch_checked([(f"{OFFICIAL}/{f}", os.path.join(VALIDATOR_DIR, *f.split("/")), h) for f, h in VALIDATOR.items()],
                  "validator files")
    code, lines = run([os.path.join(VALIDATOR_DIR, "evaluation", "validate_submission.py"), target])
    valid = code == 0
    say(f"validator: {'PASSED' if valid else 'FAILED (see the lines above)'}")

    step("8/8 Result")
    got = sha256(target)
    want = plan.get("sha256")
    say(f"file      {target.replace(data, '<data>')}")
    say(f"sha256    {got}")
    say(f"expected  {want or '(none recorded in submission.json yet)'}")
    same_file = bool(want) and got == want
    if same_model is not None:
        say(f"model fingerprints: {'same as the build Mac' if same_model else 'DIFFERENT from the build Mac'}")
    say("PASS: this computer regenerates exactly the submitted file." if same_file and valid else
        "NOT THE SAME FILE: see the lines above." if want else "No expected hash to compare with yet.")
    say("The CSV is about the competition match: keep it to yourself. This result text is safe to share.")
    write_result()
    print(f"\nSaved to: {RESULT}")
    sys.exit(0 if same_file and valid else 2)

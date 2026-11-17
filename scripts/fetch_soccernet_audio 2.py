"""Fetch the sound of some SoccerNet matches, to build and test the whistle detector on. Run it yourself.

SoccerNet (soccer-net.org) has 500 broadcast matches with their TV sound and timed events (fouls,
offsides, kick-offs, ball out of play). The labels are free. The videos are gated on Hugging Face
(huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ): request access there with your own account;
once approved, log in on this computer with a read token and SoccerNet's package downloads from
there (the old password is no longer used). These are other matches, which the competition's rules
allow; this script never touches the competition's data.

    python3 -m venv ~/soccernet-env              # its own environment, so the pipeline's pins stay untouched
    ~/soccernet-env/bin/python -m pip install SoccerNet imageio-ffmpeg==0.8.1 numpy
    ~/soccernet-env/bin/hf auth login           # paste your Hugging Face read token when it asks
    ~/soccernet-env/bin/python scripts/fetch_soccernet_audio.py --league spain_laliga --games 10

For each match it downloads the two half videos and Labels-v2.json, keeps only the sound as FLAC
(48 kHz stereo, about 300 MB a half) and deletes the video unless --keep-video is given. Matches are
taken in alphabetical order, so the same command always fetches the same matches. Already fetched
halves are skipped, so it can be stopped and started again.

Where: --dest, default ~/soccernet, outside this project. Do not put SoccerNet's videos or sound in
this project or in any repository, and do not show them to anyone: access is personal and the
agreement does not allow passing files, clips or anything derived from them on.
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import audio  # noqa: E402


MIN_FLAC_BYTES = 50_000_000       # a 45-minute half of real sound is 200 MB or more


def soccernet():
    try:
        from SoccerNet.Downloader import SoccerNetDownloader  # type: ignore
        from SoccerNet.utils import getListGames  # type: ignore
    except ImportError:
        sys.exit("SoccerNet's package is not installed: python3 -m pip install SoccerNet")
    return SoccerNetDownloader, getListGames


def download(downloader, game, files, split):
    try:
        downloader.downloadGame(game=game, files=files, spl=split, verbose=False)
    except TypeError:                                       # older versions of the package
        downloader.downloadGame(game=game, files=files, spl=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=os.path.expanduser("~/soccernet"))
    parser.add_argument("--league", default="spain_laliga", help="folder prefix, e.g. spain_laliga, england_epl")
    parser.add_argument("--split", default="train", choices=("train", "valid", "test", "challenge"))
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--quality", default="224p", choices=("224p", "720p"))
    parser.add_argument("--keep-video", action="store_true")
    args = parser.parse_args()

    Downloader, list_games = soccernet()
    ffmpeg = audio.find_ffmpeg()
    games = sorted(g for g in list_games(split=args.split) if g.startswith(args.league))[: args.games]
    if not games:
        sys.exit(f"no {args.split} matches start with {args.league!r}")
    downloader = Downloader(LocalDirectory=args.dest)
    if os.environ.get("SOCCERNET_PASSWORD"):                # only for SoccerNet's old download source
        downloader.password = os.environ["SOCCERNET_PASSWORD"]
    print(f"{len(games)} matches from {args.league} ({args.split}), into {args.dest}")
    done = 0
    for game in games:
        folder = os.path.join(args.dest, game)
        videos = [f"{half}_{args.quality}.mkv" for half in (1, 2)]
        wanted = ["Labels-v2.json"] + [v for v, half in zip(videos, (1, 2))
                                       if not os.path.exists(os.path.join(folder, f"{half}_audio.flac"))]
        try:
            download(downloader, game, wanted, args.split)
        except Exception as problem:                           # the package raises several kinds
            print(f"  {game}: download failed ({problem}). Has Hugging Face approved your access, and did"
                  " you run 'hf auth login'?")
            continue
        for half, video in zip((1, 2), videos):
            source, target = os.path.join(folder, video), os.path.join(folder, f"{half}_audio.flac")
            if os.path.exists(target) or not os.path.exists(source):
                continue
            try:
                audio.to_flac(source, target, ffmpeg)
            except RuntimeError as problem:
                print(f"  {game} half {half}: {problem}"
                      + ("  (try --quality 720p)" if args.quality == "224p" else ""))
                continue
            if os.path.getsize(target) < MIN_FLAC_BYTES:       # some videos carry only a second or two of sound
                os.remove(target)
                print(f"  {game} half {half}: the video's sound is missing or cut short"
                      + ("  (try --quality 720p)" if args.quality == "224p" else ""))
                if not args.keep_video:
                    os.remove(source)
                continue
            if not args.keep_video:
                os.remove(source)
        have = sum(os.path.exists(os.path.join(folder, f"{h}_audio.flac")) for h in (1, 2))
        done += have == 2
        print(f"  {game}: labels {'yes' if os.path.exists(os.path.join(folder, 'Labels-v2.json')) else 'no'}, "
              f"sound for {have} of 2 halves", flush=True)
    print(f"{done} of {len(games)} matches have the sound of both halves")

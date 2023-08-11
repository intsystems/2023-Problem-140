import argparse
from pathlib import Path

# pip install gdown
import gdown
import subprocess

def download_model(path):
    download_gdown(path, url='https://drive.google.com/file/d/1F-ITEqLEtzvSRwldsmSXLXvr695DKnWW/view?usp=sharing')

def download_times(path):
    download_gdown(path, url='https://drive.google.com/file/d/1scUmq1cy3TotOpI9myFuZ-MiHKdJQ7xD/view?usp=sharing')

def download_repo(path):
    url = "https://github.com/intsystems/2023-Problem-140.git"
    ensure_path(path)
    subprocess.run(["git", "clone", url], cwd=path)


def download_gdown(path, url):
    ensure_path(path)
    gdown.download(url, path, quiet=False, fuzzy=True)

def ensure_path(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='path to download model to')
    parser.add_argument("--times", help='path to download times to')
    parser.add_argument("--repo", help='directory to download repo to')
    args = parser.parse_args()

    if args.model:
        download_model(args.model)
    if args.times:
        download_times(args.times)
    if args.repo:
        download_repo(args.repo)

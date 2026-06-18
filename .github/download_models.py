import argparse
import csv
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


def iter_models(model_list):
    with open(model_list, newline='', encoding='utf-8') as f:
        for line_number, row in enumerate(csv.reader(f), start=1):
            if not row or not row[0].strip() or row[0].lstrip().startswith('#'):
                continue

            if len(row) < 2:
                raise ValueError(f'{model_list}:{line_number}: expected repo_id,filename[,local_filename]')

            repo_id = row[0].strip()
            filename = row[1].strip()
            local_filename = row[2].strip() if len(row) > 2 and row[2].strip() else Path(filename).name

            yield repo_id, filename, local_filename


def download_model(repo_id, filename, local_filename, model_dir, endpoint, retries):
    target = model_dir / local_filename
    if target.exists() and target.stat().st_size > 0:
        print(f'Skipping existing model: {target}')
        return

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f'Downloading {repo_id}/{filename} -> {target} (attempt {attempt}/{retries})')
            downloaded = Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                endpoint=endpoint
            ))

            if downloaded.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), target)

            return
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break

            sleep_seconds = min(60, attempt * 5)
            print(f'Download failed: {exc}. Retrying in {sleep_seconds}s...')
            time.sleep(sleep_seconds)

    raise RuntimeError(f'Failed to download {repo_id}/{filename}') from last_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-list', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--endpoint', type=str, default='https://huggingface.co')
    parser.add_argument('--retries', type=int, default=5)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, filename, local_filename in iter_models(args.model_list):
        download_model(repo_id, filename, local_filename, model_dir, args.endpoint, args.retries)

"""Download and clean the raw WikiText-2 training dataset."""

from pathlib import Path
from urllib.request import Request, urlopen

from tokenizers import Tokenizer


DATA_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/master/"
    "word_language_model/data/wikitext-2/train.txt"
)
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT_DIR / "model" / "input.txt"
TOKENIZER_PATH = ROOT_DIR / "model" / "tokenizer.json"


def download_dataset() -> tuple[int, int]:
    request = Request(DATA_URL, headers={"User-Agent": "test-llm-dataset-downloader"})
    with urlopen(request, timeout=60) as response:
        raw_text = response.read().decode("utf-8")

    cleaned_text = "\n".join(
        line for line in raw_text.splitlines() if line.strip()
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(cleaned_text, encoding="utf-8")

    if TOKENIZER_PATH.is_file():
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        estimated_bpe_tokens = len(tokenizer.encode(cleaned_text).ids)
    else:
        estimated_bpe_tokens = 0
        print(
            "Warning: model/tokenizer.json was not found; "
            "BPE token count could not be estimated."
        )

    return len(cleaned_text), estimated_bpe_tokens


if __name__ == "__main__":
    character_count, bpe_token_count = download_dataset()
    print(f"Saved cleaned WikiText-2 data to {OUTPUT_PATH}")
    print(f"Character count: {character_count:,}")
    print(f"Estimated BPE token count: {bpe_token_count:,}")

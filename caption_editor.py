"""Caption correction and polishing tool.

Installation:
    pip install pysrt pycorrector opencc-python-reimplemented textblob tqdm pyyaml
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable

import pysrt
import pycorrector
import yaml
from opencc import OpenCC
from textblob import TextBlob
from tqdm import tqdm


def detect_format(text: str) -> str:
    """Return 'srt' if text contains SRT timestamps else 'text'."""
    pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} -->")
    return "srt" if pattern.search(text) else "text"


def load_special_terms(path: Path) -> Dict[str, str]:
    """Load mapping from a JSON or YAML configuration file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def replace_terms(text: str, terms: Dict[str, str]) -> str:
    for src, dst in terms.items():
        text = text.replace(src, dst)
    return text


def correct_spelling(text: str) -> str:
    corrected, _ = pycorrector.correct(text)
    blob = TextBlob(corrected)
    return str(blob.correct())


def convert_simplified_to_traditional(text: str, converter: OpenCC) -> str:
    return converter.convert(text)


def process_text(text: str, terms: Dict[str, str], converter: OpenCC) -> str:
    text = replace_terms(text, terms)
    text = correct_spelling(text)
    text = convert_simplified_to_traditional(text, converter)
    return text


def process_srt(input_path: Path, output_path: Path, terms: Dict[str, str], logger: logging.Logger):
    subs = pysrt.open(str(input_path), encoding="utf-8")
    converter = OpenCC("s2t")
    for sub in tqdm(subs, desc=str(input_path)):
        sub.text = process_text(sub.text, terms, converter)
    subs.save(str(output_path), encoding="utf-8")
    logger.info(f"Processed {input_path} -> {output_path}")


def process_txt(input_path: Path, output_path: Path, terms: Dict[str, str], logger: logging.Logger):
    converter = OpenCC("s2t")
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=str(input_path)):
            processed = process_text(line.rstrip("\n"), terms, converter)
            fout.write(processed + "\n")
    logger.info(f"Processed {input_path} -> {output_path}")


def iter_subtitle_files(directory: Path) -> Iterable[Path]:
    for ext in ("*.srt", "*.txt"):
        for path in directory.glob(ext):
            yield path


def main():
    parser = argparse.ArgumentParser(description="Subtitle correction tool")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--special-terms-config")
    parser.add_argument("--batch-dir")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("caption-editor")

    terms = load_special_terms(Path(args.special_terms_config)) if args.special_terms_config else {}

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        for path in iter_subtitle_files(batch_dir):
            out_path = Path(args.output) / path.name if args.output else path
            with open(path, "r", encoding="utf-8") as f:
                first_chunk = f.read(2000)
            fmt = detect_format(first_chunk)
            if fmt == "srt":
                process_srt(path, out_path, terms, logger)
            else:
                process_txt(path, out_path, terms, logger)
    elif args.input and args.output:
        input_path = Path(args.input)
        output_path = Path(args.output)
        with open(input_path, "r", encoding="utf-8") as f:
            first_chunk = f.read(2000)
        fmt = detect_format(first_chunk)
        if fmt == "srt":
            process_srt(input_path, output_path, terms, logger)
        else:
            process_txt(input_path, output_path, terms, logger)
    else:
        parser.error("Provide --input & --output or --batch-dir")


if __name__ == "__main__":
    main()

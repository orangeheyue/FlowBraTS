import argparse
import json

from src.config import load_config
from src.models.m2f_adapter import M2FTransAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = M2FTransAdapter(
        config["m2ftrans"]["root"], config["m2ftrans"]["checkpoint"]
    )
    print(json.dumps(adapter.load_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

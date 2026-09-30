import argparse
import inspect
import sys
from pathlib import Path

try:
    this_file = Path(__file__).resolve()
except NameError:
    this_file = Path(inspect.currentframe().f_code.co_filename).resolve()

sys.path.insert(0, str(this_file.parent.parent.parent))

from src.etl.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()

    spec_path = this_file.parent.parent.parent / args.spec
    manifest = run_pipeline(spec_path, catalog=args.catalog, schema=args.schema)
    print(manifest)


if __name__ == "__main__":
    main()

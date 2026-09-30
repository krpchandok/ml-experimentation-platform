from pathlib import Path
from dotenv import load_dotenv
from sklearn.datasets import load_iris
import io
import sys

HEADER = "sepal_length,sepal_width,petal_length,petal_width,species"
SPECIES_NAMES = ["setosa", "versicolor", "virginica"]


def build_dirty_iris_csv():
    data = load_iris()
    rows = []

    for features, target in zip(data.data, data.target):
        rows.append(",".join([*(str(v) for v in features), SPECIES_NAMES[target]]))

    dirty_rows = [
        ",".join(["", "3.0", "1.4", "0.2", "setosa"]),
        ",".join(["5.0", "", "1.4", "0.2", "setosa"]),
        ",".join(["abc", "3.0", "1.4", "0.2", "setosa"]),
        ",".join(["999", "3.0", "1.4", "0.2", "setosa"]),
        ",".join(["5.0", "3.0", "1.4", "0.2", "unknown_flower"]),
        rows[0],
        rows[0],
    ]

    return "\n".join([HEADER, *rows, *dirty_rows]) + "\n"


def upload_to_volume(csv_text, volume_path="/Volumes/workspace/ml_platform/raw/iris.csv"):
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    w.files.upload(volume_path, io.BytesIO(csv_text.encode("utf-8")), overwrite=True)
    return volume_path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    csv_text = build_dirty_iris_csv()
    path = upload_to_volume(csv_text)
    row_count = csv_text.strip().count(chr(10))
    print(f"Uploaded {row_count} rows to {path}")


if __name__ == "__main__":
    main()

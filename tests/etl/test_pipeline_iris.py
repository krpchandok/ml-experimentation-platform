from src.etl.spec import load_spec
from src.etl.transforms import read_raw, transform
from src.etl.validation import validate
from scripts.seed_raw_iris import build_dirty_iris_csv


def test_dirty_iris_validation_counts(spark, tmp_path):
    spec = load_spec("configs/datasets/iris.yaml")
    csv_path = tmp_path / "iris.csv"
    csv_path.write_text(build_dirty_iris_csv())
    spec.raw_path = str(csv_path)

    raw_df = read_raw(spark, spec)
    clean_df, quarantine_df = validate(raw_df, spec)
    clean_df = transform(clean_df, spec)

    rows_in = raw_df.count()
    rows_rejected = quarantine_df.count()
    rows_out = clean_df.count()

    assert rows_in == 157
    assert rows_rejected == 8
    assert rows_out == 149
    assert rows_in == rows_out + rows_rejected

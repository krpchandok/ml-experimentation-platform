from pyspark.sql.types import StructType, StructField, StringType

from src.etl.spec import DatasetSpec, Column
from src.etl.validation import validate


def make_spec():
    return DatasetSpec(
        name="test",
        raw_path="unused",
        format="csv",
        columns=[
            Column(name="a", type="double", required=True, min=0, max=10),
            Column(name="label", type="string", required=True),
        ],
        label_column="label",
        allowed_labels=["x", "y"],
        max_rejection_rate=1.0,
    )


def make_df(spark, rows):
    schema = StructType([StructField("a", StringType(), True), StructField("label", StringType(), True)])
    return spark.createDataFrame(rows, schema=schema)


def test_null_required_column(spark):
    spec = make_spec()
    df = make_df(spark, [("", "x")])

    clean, quarantine = validate(df, spec)

    assert clean.count() == 0
    assert quarantine.count() == 1
    assert "null_a" in quarantine.collect()[0].rejection_reason


def test_cast_failure(spark):
    spec = make_spec()
    df = make_df(spark, [("abc", "x")])

    clean, quarantine = validate(df, spec)

    assert clean.count() == 0
    assert quarantine.count() == 1
    assert "cast_failure_a" in quarantine.collect()[0].rejection_reason


def test_out_of_range(spark):
    spec = make_spec()
    df = make_df(spark, [("999", "x")])

    clean, quarantine = validate(df, spec)

    assert clean.count() == 0
    assert quarantine.count() == 1
    assert "out_of_range_a" in quarantine.collect()[0].rejection_reason


def test_unknown_label(spark):
    spec = make_spec()
    df = make_df(spark, [("5", "unknown")])

    clean, quarantine = validate(df, spec)

    assert clean.count() == 0
    assert quarantine.count() == 1
    assert "unknown_label" in quarantine.collect()[0].rejection_reason


def test_duplicate(spark):
    spec = make_spec()
    df = make_df(spark, [("5", "x"), ("5", "x")])

    clean, quarantine = validate(df, spec)

    assert clean.count() == 1
    assert quarantine.count() == 1
    assert "duplicate" in quarantine.collect()[0].rejection_reason


def test_clean_row_passes(spark):
    spec = make_spec()
    df = make_df(spark, [("5", "x")])

    clean, quarantine = validate(df, spec)

    assert clean.count() == 1
    assert quarantine.count() == 0

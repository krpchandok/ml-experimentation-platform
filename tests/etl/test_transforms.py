from pyspark.sql.types import StructType, StructField, StringType

from src.etl.spec import DatasetSpec, Column
from src.etl.transforms import build_raw_schema, read_raw, transform


def make_spec():
    return DatasetSpec(
        name="test",
        raw_path="unused",
        format="csv",
        columns=[
            Column(name="a", type="double", required=True),
            Column(name="label", type="string", required=True),
        ],
        label_column="label",
        allowed_labels=["x"],
        max_rejection_rate=1.0,
    )


def test_build_raw_schema_is_all_strings():
    spec = make_spec()
    schema = build_raw_schema(spec)

    assert [f.name for f in schema.fields] == ["a", "label"]
    assert all(f.dataType.typeName() == "string" for f in schema.fields)


def test_transform_trims_and_casts(spark):
    spec = make_spec()
    schema = StructType([StructField("a", StringType(), True), StructField("label", StringType(), True)])
    df = spark.createDataFrame([("  5.0  ", "  x  ")], schema=schema)

    result = transform(df, spec)
    row = result.collect()[0]

    assert row.a == 5.0
    assert row.label == "x"


def test_read_raw_uses_explicit_schema_no_inference(spark, tmp_path):
    spec = make_spec()
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("a,label\n5,x\n")
    spec.raw_path = str(csv_path)

    df = read_raw(spark, spec)

    assert {f.name: f.dataType.typeName() for f in df.schema.fields} == {"a": "string", "label": "string"}

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


def build_raw_schema(spec):
    return StructType([StructField(c.name, StringType(), True) for c in spec.columns])


def read_raw(spark, spec):
    return (
        spark.read.format(spec.format)
        .schema(build_raw_schema(spec))
        .option("header", True)
        .load(spec.raw_path)
    )


def transform(df, spec):
    result = df
    for col in spec.columns:
        c = F.trim(F.col(col.name))
        if col.type in ("double", "int", "long"):
            result = result.withColumn(col.name, c.try_cast(col.type))
        else:
            result = result.withColumn(col.name, c)
    return result

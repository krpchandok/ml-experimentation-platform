from pyspark.sql import functions as F
from pyspark.sql.window import Window


def validate(df, spec):
    reason_exprs = []

    for col in spec.columns:
        c = F.col(col.name)

        if col.required:
            reason_exprs.append(
                F.when(c.isNull() | (F.trim(c) == ""), F.lit(f"null_{col.name}"))
            )

        if col.type in ("double", "int", "long"):
            casted = F.trim(c).try_cast(col.type)
            reason_exprs.append(
                F.when(
                    c.isNotNull() & (F.trim(c) != "") & casted.isNull(),
                    F.lit(f"cast_failure_{col.name}"),
                )
            )
            if col.min is not None:
                reason_exprs.append(
                    F.when(casted.isNotNull() & (casted < col.min), F.lit(f"out_of_range_{col.name}"))
                )
            if col.max is not None:
                reason_exprs.append(
                    F.when(casted.isNotNull() & (casted > col.max), F.lit(f"out_of_range_{col.name}"))
                )

    reason_exprs.append(
        F.when(~F.col(spec.label_column).isin(spec.allowed_labels), F.lit("unknown_label"))
    )

    raw_columns = [col.name for col in spec.columns]
    dup_window = Window.partitionBy(*raw_columns).orderBy(F.monotonically_increasing_id())

    df = df.withColumn("_dup_rank", F.row_number().over(dup_window))
    reason_exprs.append(F.when(F.col("_dup_rank") > 1, F.lit("duplicate")))

    reasons_array = F.array(*[F.coalesce(e, F.lit("")) for e in reason_exprs])
    df = df.withColumn("_reasons", F.array_remove(reasons_array, ""))
    df = df.withColumn("rejection_reason", F.concat_ws(";", F.col("_reasons")))

    quarantine_df = df.filter(F.size(F.col("_reasons")) > 0).drop("_dup_rank", "_reasons")
    clean_df = df.filter(F.size(F.col("_reasons")) == 0).drop("_dup_rank", "_reasons", "rejection_reason")

    return clean_df, quarantine_df

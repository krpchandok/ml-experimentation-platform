from dataclasses import dataclass
import yaml


@dataclass
class Column:
    name: str
    type: str
    required: bool = False
    min: float = None
    max: float = None


@dataclass
class DatasetSpec:
    name: str
    raw_path: str
    format: str
    columns: list
    label_column: str
    allowed_labels: list
    max_rejection_rate: float


def load_spec(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    columns = [Column(**c) for c in data["columns"]]

    return DatasetSpec(
        name=data["name"],
        raw_path=data["raw_path"],
        format=data.get("format", "csv"),
        columns=columns,
        label_column=data["label_column"],
        allowed_labels=data["allowed_labels"],
        max_rejection_rate=data.get("max_rejection_rate", 0.1),
    )

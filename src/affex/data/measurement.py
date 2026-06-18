from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RT = 1.987204258 * 0.001 * 298


@dataclass
class ExactMeasurement:
    value: float

    def to_csv_dict(self):
        return {"kd_type": "exact", "kd_value": self.value, "kd_lower": None, "kd_upper": None}


@dataclass
class BoundedMeasurement:
    bound: float
    bound_type: Literal[">", "<"]

    def to_csv_dict(self):
        return {"kd_type": self.bound_type, "kd_value": self.bound, "kd_lower": None, "kd_upper": None}


@dataclass
class IntervalMeasurement:
    estimate: float
    lower_bound: float
    upper_bound: float

    def to_csv_dict(self):
        return {
            "kd_type": "interval",
            "kd_value": self.estimate,
            "kd_lower": self.lower_bound,
            "kd_upper": self.upper_bound,
        }


Measurement = ExactMeasurement | BoundedMeasurement | IntervalMeasurement


def measurement_from_dict(csv_row: dict[str, Any]) -> Measurement:
    if csv_row["kd_type"] == "exact":
        return ExactMeasurement(csv_row["kd_value"])

    elif csv_row["kd_type"] == "interval":
        return IntervalMeasurement(
            estimate=csv_row["kd_value"],
            lower_bound=csv_row["kd_lower"],
            upper_bound=csv_row["kd_upper"],
        )
    elif csv_row["kd_type"] in (">", "<"):
        return BoundedMeasurement(bound=csv_row["kd_value"], bound_type=csv_row["kd_type"])
    else:
        raise ValueError(f"Could not parse Measurement for {csv_row}")


@dataclass
class DataItem:
    uid: str
    pdb: Path
    receptor_chains: list[str]
    ligand_chains: list[str]
    affinity: Measurement

    def to_csv_dict(self):
        return {
            "uid": self.uid,
            "pdb": str(self.pdb),
            "receptor_chains": "".join(self.receptor_chains),
            "ligand_chains": "".join(self.ligand_chains),
            **self.affinity.to_csv_dict(),
        }

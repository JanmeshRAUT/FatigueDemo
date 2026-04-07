from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class DataConfig:
    dataset_path: Path = Path("data/fatigue_dataset_80_accuracy.csv")
    target_col: str = "label"
    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42


@dataclass
class FeatureConfig:
    # CV + IoT features for hybrid fatigue prediction.
    features: tuple[str, ...] = (
        "EAR",
        "MAR",
        "PERCLOS",
        "blink_rate",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "heart_rate",
        "spo2",
        "temperature",
    )


@dataclass
class RFConfig:
    epochs: int = 10
    trees_per_epoch: int = 20
    early_stopping_patience: int = 3
    min_delta: float = 0.0005
    max_depth: int = 12
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    random_state: int = 42


@dataclass
class RFTuneConfig:
    enabled: bool = True
    max_depth_values: tuple[int, ...] = (10, 12, 14)
    min_samples_split_values: tuple[int, ...] = (2, 4)
    min_samples_leaf_values: tuple[int, ...] = (1, 2)


@dataclass
class XGBConfig:
    epochs: int = 120  # n_estimators acts like boosting epochs
    early_stopping_rounds: int = 20
    max_depth: int = 6
    learning_rate: float = 0.08
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.2
    reg_lambda: float = 1.5
    min_child_weight: int = 3
    random_state: int = 42


@dataclass
class XGBTuneConfig:
    enabled: bool = True
    max_depth_values: tuple[int, ...] = (4, 6)
    learning_rate_values: tuple[float, ...] = (0.05, 0.08)
    min_child_weight_values: tuple[int, ...] = (1, 3)


@dataclass
class EnsembleConfig:
    enabled: bool = True
    alpha_values: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass
class OutputConfig:
    root_dir: Path = Path("outputs")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    rf_tune: RFTuneConfig = field(default_factory=RFTuneConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    xgb_tune: XGBTuneConfig = field(default_factory=XGBTuneConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

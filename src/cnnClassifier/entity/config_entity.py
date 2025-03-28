#Entity is the return type of any function(say here data ingestion).
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True) #To access this as class variable from other files we use dataclass(entity)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
from dataclasses import dataclass


# ---------------------- Data Ingestion ----------------------
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


# ---------------------- Data Validation ----------------------
@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    drift_report_file_path: str


# ---------------------- Data Transformation ----------------------
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


# ---------------------- Metrics ----------------------
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


# ---------------------- Model Trainer ----------------------
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact


# ---------------------- Model Evaluation ----------------------
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_score: float
    s3_model_path: str
    trained_model_path: str


# ---------------------- Model Pusher ----------------------
@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    trained_model_path: str
from .arima_pipeline import ARIMAPipeline
from .base import ModelPipeline
from .bocpd_pipeline import BOCPDPipeline
from .mmm_pipeline import MMMPipeline
from .psi_pipeline import PSIPipeline

# Registry: model_name (as used in model_routing.yaml) → pipeline class
PIPELINE_REGISTRY: dict[str, type[ModelPipeline]] = {
    "BOCPD": BOCPDPipeline,
    "MMM": MMMPipeline,
    "PSI": PSIPipeline,
    "ARIMA": ARIMAPipeline,
}

__all__ = [
    "ModelPipeline",
    "BOCPDPipeline",
    "MMMPipeline",
    "PSIPipeline",
    "ARIMAPipeline",
    "PIPELINE_REGISTRY",
]

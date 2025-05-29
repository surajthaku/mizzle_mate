from app.models.ci_cd import CICDModel
from app.models.instance_creation import InstanceCreationModel
from app.models.storage import StorageModel
from app.models.logs import LogsModel
from app.models.monitoring import MonitoringModel

MODULE_REGISTRY = {
    "ci_cd": CICDModel,
    "instance_creation": InstanceCreationModel,
    "storage": StorageModel,
    "logs": LogsModel,
    "monitoring": MonitoringModel
}

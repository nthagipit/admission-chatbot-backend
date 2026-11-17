from enum import StrEnum

class StatusDocumentEnum(StrEnum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
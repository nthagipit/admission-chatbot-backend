from enum import StrEnum

class AuthProviderEnum(StrEnum):
    LOCAL = "local"
    GOOGLE = "google"

class RoleEnum(StrEnum):
    ADMIN = "admin"
    USER = "user"

class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
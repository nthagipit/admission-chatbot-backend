class AppException(Exception):
    def __init__(self, status_code: int, message: str, headers: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.headers = headers
        super().__init__(message)
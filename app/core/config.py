import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_USER: str = os.getenv("USER")
    DB_PASSWORD: str = os.getenv("PASSWORD")
    DB_HOST: str = os.getenv("HOST")
    DB_PORT: str = os.getenv("PORT")
    DB_NAME: str = os.getenv("DBNAME")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    REFRESH_SECRET_KEY: str = os.getenv("REFRESH_SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE: int = int(os.getenv("ACCESS_TOKEN_EXPIRE"))
    REFRESH_TOKEN_EXPIRE: int = int(os.getenv("REFRESH_TOKEN_EXPIRE"))
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI: str = os.getenv("GOOGLE_REDIRECT_URI")
    SESSION_SECRET_KEY: str = os.getenv("SESSION_SECRET_KEY")
    NEON_POSTGRESQL: str = os.getenv("NEON_POSTGRESQL")
    


settings = Settings()
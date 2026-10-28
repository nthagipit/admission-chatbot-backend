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


settings = Settings()
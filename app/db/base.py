from sqlalchemy import create_engine
from sqlalchemy.orm import  sessionmaker
from app.core.config import settings

DATABASE_URL =f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

DATABASE_URL_NEON = settings.NEON_POSTGRESQL

engine = create_engine(DATABASE_URL_NEON)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.rollback()
        raise
    finally:
        db.close()
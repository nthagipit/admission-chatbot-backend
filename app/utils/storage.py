
from app.core.exception import AppException
import logging
from app.core.supabase import supabase
from app.core.config import settings

logger = logging.getLogger("app.utils.storage")

def upload_file_to_supabase(file_bytes: bytes, file_name: str, content_type: str = "application/pdf"):
    bucket = settings.SUPABASE_BUCKET_NAME

    try:
        res = supabase.storage.from_(bucket).upload(
            path=file_name,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )

        public_url = supabase.storage.from_(bucket).get_public_url(file_name)
        return public_url
    except Exception as e:
        logger.error(f"Supabase upload error: {str(e)}", exc_info=True)
        raise AppException(status_code=500, message=f"Supabase upload error: {str(e)}")

def download_file_from_supabase(file_name: str):
    bucket = settings.SUPABASE_BUCKET_NAME
    try:
        return supabase.storage.from_(bucket).download(file_name)
    except Exception as e:
        logger.error(f"Supabase download error: {str(e)}", exc_info=True)
        raise AppException(status_code=500, message=f"Couldn't download file from Supabase Storage: {str(e)}")

def delete_file_from_supabase(file_name: str) -> bool:
    bucket = settings.SUPABASE_BUCKET_NAME
    try:
        res = supabase.storage.from_(bucket).remove([file_name])
        return len(res) > 0
    except Exception as e:
        logger.error(f"Supabase delete error: {str(e)}", exc_info=True)
        return False
from supabase import create_client, Client
from app.core.exception import AppException
from app.core.config import settings

if not settings.SUPABASE_URL or not settings.SUPABASE_SECRET_KEY:
    raise AppException(status_code=500, message="Supabase URL or Key is not configured in .env")

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SECRET_KEY)
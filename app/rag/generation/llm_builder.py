from app.core.config import settings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


llm_builder = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash", 
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
    )

llm_fast_builder = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.2,
    thinking_level="minimal",
    max_retries=1,
    timeout=30,
)
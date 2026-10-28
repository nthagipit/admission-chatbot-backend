from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

candidate_models = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]

print("Đang kiểm tra API Key với các model Gemini...")
working_model = None

for m in candidate_models:
    try:
        print(f"- Thử model: '{m}' ...", end=" ", flush=True)
        llm = ChatGoogleGenerativeAI(
            model=m,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1
        )
        resp = llm.invoke("Hi")
        print("-> [HOẠT ĐỘNG TỐT]")
        working_model = m
        break
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg:
            print("-> [404 Not Found]")
        elif "429" in err_msg:
            print("-> [429 Quota Exceeded]")
        else:
            print(f"-> [Lỗi: {type(e).__name__}]")

print(f"\nModel được chọn: {working_model}")

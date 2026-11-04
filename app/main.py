from app.core.config import settings
from starlette.middleware.sessions import SessionMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.schemas.base_schema import DataResponse
from app.core.exception import AppException
from fastapi import FastAPI, Request,status
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router as v1_router


origins = [
    "http://localhost:3000",
    settings.CORS_ORIGINS
]


app = FastAPI(
    title="Admissions Chatbot API",
    version="1.0.0"
)

app.include_router(v1_router, prefix="/api/v1")
app.add_middleware(
    SessionMiddleware, 
    secret_key=settings.SESSION_SECRET_KEY,
    same_site="lax",
    https_only=True
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppException)
def handle_app_exception(request, exc):
    response_body = DataResponse.error(status_code=exc.status_code, detail=exc.message)
    if hasattr(response_body, "model_dump"):
        content = response_body.model_dump(exclude_none=True)  # Pydantic v2
    else:
        content = response_body.dict(exclude_none=True)
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    custom_errors = []
    
    for error in exc.errors():
        field = error.get("loc")[-1] if error.get("loc") else "unknown"
        original_msg = error.get("msg", "")
        custom_errors.append({
            "field": field,
            "message": original_msg
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Dữ liệu đầu vào không hợp lệ",
            "errors": custom_errors
        }
    )




@app.get("/")
async def root():
    return {"message": "Admissions Chatbot API is running"}
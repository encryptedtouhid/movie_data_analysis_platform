from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.api.routes import health

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A comprehensive movie data analysis platform with REST API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health.router,
    prefix=settings.api_v1_prefix,
    tags=["Health"],
)


@app.on_event("startup")
async def startup_event() -> None:
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"API docs available at: http://{settings.host}:{settings.port}/docs")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    print(f"Shutting down {settings.app_name}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

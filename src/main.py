from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.api.routes import health, data_processing
from src.utils.logger import get_logger

logger = get_logger("main_app", "core")

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

logger.info("Registering API routers")
app.include_router(
    health.router,
    prefix=settings.api_v1_prefix,
    tags=["Health"],
)

app.include_router(
    data_processing.router,
    prefix=settings.api_v1_prefix + "/dataprocess",
    tags=["Data Processing"],
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=" * 50)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"API docs available at: http://{settings.host}:{settings.port}/docs")
    logger.info(f"ReDoc available at: http://{settings.host}:{settings.port}/redoc")
    logger.info(f"Health check: http://{settings.host}:{settings.port}{settings.api_v1_prefix}/health")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info(f"Shutting down {settings.app_name}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

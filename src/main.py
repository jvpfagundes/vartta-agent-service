import uvicorn

from src.infrastructure.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.infrastructure.entrypoints.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )

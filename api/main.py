"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.core.exceptions import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from api.routers import auth, history, images, predict, process
from api.schemas.common import ApiResponse
from api.services.model_service import load_models

# ---------------------------------------------------------------------------
# Lifespan — runs before the first request
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    # cleanup (if needed) goes here


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Diatoms Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Cloudflare Pages + local dev; update origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        # Add Cloudflare Pages URL here after Phase 5, e.g.:
        # "https://diatoms.pages.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Routers
app.include_router(auth.router, prefix="/api")
app.include_router(process.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(history.router, prefix="/api")


# ---------------------------------------------------------------------------
# Health check (no auth required — used by UptimeRobot / cron keep-alive)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
async def health():
    return ApiResponse(message="API is healthy", data={"status": "ok"})

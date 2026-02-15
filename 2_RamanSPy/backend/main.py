from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import data, preprocess, analyze
from app.core import config
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RamanSPy API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(preprocess.router, prefix="/api/preprocess", tags=["preprocess"])
app.include_router(analyze.router, prefix="/api/analyze", tags=["analyze"])

@app.get("/")
def health_check():
    return {"status": "ok"}

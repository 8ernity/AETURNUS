import os
import sys
import uvicorn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("  AutoHeal Fleet: Predictive Maintenance for Connected Fleet")
    print("  Tata Elxsi TELIPORT Season 4 · Case Study 1")
    print("=" * 60)
    print("Starting FastAPI backend and Web Dashboard...")
    print("  Local URL:  http://127.0.0.1:8000")
    print("  API Docs:   http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()

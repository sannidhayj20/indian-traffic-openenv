# server/app.py — root-level entry point for openenv validator
import uvicorn
from indian_traffic_env.server.app import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
from openenv.core.env_server import create_fastapi_app
from indian_traffic_env.models import TrafficAction, TrafficObservation
from indian_traffic_env.server.environment import TrafficEnvironment
import uvicorn

app = create_fastapi_app(TrafficEnvironment, TrafficAction, TrafficObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
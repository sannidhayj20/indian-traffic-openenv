from openenv.core.env_client import EnvClient
from indian_traffic_env.models import TrafficAction, TrafficObservation, TrafficState

class TrafficEnv(EnvClient):
    """Client for the Indian Traffic Signal environment."""
    action_type = TrafficAction
    observation_type = TrafficObservation
    state_type = TrafficState
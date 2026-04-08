# Start from Python 3.11 (a standard base image — think of it as a clean Linux machine)
FROM python:3.11-slim

# Set the working folder inside the container
WORKDIR /app

# Copy your dependency list first (so Docker can cache this layer)
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir openenv-core fastapi uvicorn[standard] pydantic

# Copy all your code into the container
COPY indian_traffic_env/ ./indian_traffic_env/
COPY traffic_data.py .

# Tell Docker which port the server listens on
EXPOSE 7860

# The command that runs when someone does `docker run`
CMD ["uvicorn", "indian_traffic_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]
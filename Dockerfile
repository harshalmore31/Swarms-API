FROM python:3.11-slim

# Set environment variables to ensure Python output is logged and bytecode is not written
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy the requirements file first for dependency caching
COPY api/requirements.txt .


# Install build dependencies, then install Python dependencies, and finally remove build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get purge -y --auto-remove gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the API source code into the container
COPY api/ .

# Create a non-root user and change ownership of the application folder
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser /app
USER appuser

# Expose port 80 for the application
EXPOSE 8080

# Create a startup script within the Dockerfile
RUN echo '#!/bin/bash\n\
# Calculate optimal number of workers\n\
# Common formula: (2 * CPU cores) + 1\n\
CORES=$(nproc)\n\
WORKERS=$((CORES * 2 + 1))\n\
\n\
echo "Starting with $WORKERS workers on $CORES CPU cores"\n\
\n\
# Start Gunicorn with production settings\n\
exec gunicorn api:app \\\n\
  --workers=$WORKERS \\\n\
  --worker-class=uvicorn.workers.UvicornWorker \\\n\
  --bind=0.0.0.0:8080 \\\n\
  --timeout=120 \\\n\
  --keepalive=65 \\\n\
  --max-requests=1000 \\\n\
  --max-requests-jitter=50 \\\n\
  --graceful-timeout=30 \\\n\
  --log-level=info \\\n\
  --access-logfile=- \\\n\
  --error-logfile=- \\\n\
  --worker-tmp-dir=/dev/shm \\\n\
  --preload\n\
' > /start.sh && chmod +x /start.sh

# Use the startup script as the entry point
CMD ["/start.sh"]
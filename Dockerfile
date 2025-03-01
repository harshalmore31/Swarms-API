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

# Start the API using Uvicorn
CMD ["sh", "-c", "gunicorn api:app -w $(python -c 'import multiprocessing as mp; print(mp.cpu_count() * 2 + 1)') -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080"]

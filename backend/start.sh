#!/bin/bash
set -e

echo "=== AutoCognitix Backend Startup ==="

# Run database migrations
echo "Running Alembic migrations..."
python -m alembic upgrade head
echo "Migrations complete!"

# Start the server
echo "Starting uvicorn server on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4

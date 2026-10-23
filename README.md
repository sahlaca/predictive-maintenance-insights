# Predictive Maintenance Insights

Isolated client-facing prototype using the requested stack:

- **Frontend:** Next.js + TypeScript
- **Backend:** FastAPI serving a Random Forest model
- **Reverse proxy:** Nginx routes `/api/*` to FastAPI and all other requests to Next.js
- **Local orchestration:** Docker Compose

> **Prototype only:** this uses a sample CSV and is not connected to live equipment. Predictions are not calibrated risk probabilities and must not be used for safety-critical or operational decisions.

## Run with Docker Compose

From this folder, start the services with `docker compose up --build`. Open `http://localhost:8081` in a browser. FastAPI documentation is available at `http://localhost:8081/api/docs`.

This is the recommended way to demo because it starts the API, web frontend, and Nginx proxy together. The first build downloads dependencies and may take a few minutes.

## Run without Docker

Use Python 3.10+ and Node.js 20+.

1. In one terminal, create a Python virtual environment, install `backend/requirements.txt`, then run `uvicorn app.main:app --app-dir backend --reload --port 8000` from this folder.
2. In another terminal, run `cd frontend && npm install && npm run dev`.
3. Open `http://localhost:3000`. Next.js forwards `/api/*` to the local FastAPI server.

To run API tests: `cd backend && pytest` (after installing backend requirements).

## Deploy

Build the three containers with Docker Compose on a host that supports Docker, expose Nginx's port 80 through your hosting platform or ingress, and configure TLS at the platform or edge proxy. The bundled CSV is copied into the backend image. For a public client demo, consider basic access control and avoid exposing sensitive datasets.

Routes:

- `GET /api/health` — service health check
- `GET /api/metadata` — dataset and model summary
- `GET /api/metrics` — holdout metrics per failure category
- `POST /api/predict` — classify one machine reading
- `GET /api/docs` — generated API documentation

The dashboard presents plain-language operating patterns from the project presentation, ranks model feature importance, and shows the 5th–95th percentile ranges among historical no-failure records. These ranges are **descriptive references only**, not safe operating limits or input restrictions; values outside the training data are accepted but receive an extrapolation warning. Approved manufacturer specifications and site procedures take precedence.

The service trains and caches the model when the first metadata, metrics, or prediction request arrives. SMOTENC is fitted only to the training partition; evaluation uses a stratified, original-distribution holdout.

## Project layout

- `backend/app/` — API endpoints and model pipeline
- `backend/tests/` — API smoke tests
- `frontend/src/app/` — Next.js demo experience
- `nginx/default.conf` — reverse-proxy routing
- `data/predictive_maintenance.csv` — isolated copy of the sample data
- `docker-compose.yml` — local and container orchestration

All demo files and the CSV copy reside here. Original project scripts, data, and artifacts in the parent folder are not modified by the demo.

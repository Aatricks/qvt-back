# qvt-back

FastAPI backend that generates Vega-Lite JSON specs (Altair) from uploaded QVCT/QVT survey files.

## Run the API (dev)

- Health check: `GET /health`
- Supported chart keys: `GET /api/visualize/supported-keys`
- Generate a chart: `POST /api/visualize/{chart_key}` (multipart form with `hr_file` and optional `filters`/`config` JSON strings)

Typical dev command:

- `uvicorn src.api.app:app --reload --port 8000`

### CORS

If you serve the frontend from a different origin (instead of using the Vite dev proxy), you can set:

- `QVCTI_CORS_ALLOW_ORIGINS=*` (default) or a comma-separated list.

## Run the React site

The React/Vite app lives in `site/` and can load the bundled sample CSV:

- `site/public/PROJET_POV-ML_-_Fichier_de_données_brutes.csv`

Dev workflow:

- in `site/`: `npm install`
- in `site/`: `npm run dev`

The site provides 3 role-based pages:

- Employé: basic state + suggested actions
- Manager: curated insights + recommended actions
- RH: full chart explorer with filters/config
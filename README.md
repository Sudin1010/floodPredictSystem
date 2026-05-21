# Flood Prediction System

A university Flood Prediction System built with FastAPI, Jinja2 templates, PostgreSQL, Bootstrap, and an Artificial Neural Network implemented from scratch. The application supports user registration, login/logout, flood risk prediction, and a user-specific prediction history dashboard.

The ANN model is loaded from `models/ann_scratch_model.pkl`. The application uses the saved model parameters, feature order, scaling statistics, and preprocessing pipeline for inference only.

## Architecture

The project is organized into focused modules:

- `app/main.py` creates the FastAPI app, configures middleware/static files, includes routers, and runs startup database setup.
- `app/routes/` contains HTTP route handlers for authentication, prediction, and dashboard/history pages.
- `app/auth/` contains password hashing and session/user authentication helpers.
- `app/database/` contains the PostgreSQL connection setup and SQLAlchemy ORM models.
- `app/ml/` contains model loading, preprocessing, ANN forward propagation, and risk mapping.
- `app/services/` contains reusable business/database operations such as prediction history persistence.
- `app/templates/` contains Jinja2 HTML templates.
- `app/static/` contains CSS and JavaScript assets.

## Folder Structure

```text
Flood_Predictiopn_System/
├── app/
│   ├── auth/
│   │   ├── hashing.py
│   │   └── session.py
│   ├── database/
│   │   ├── connection.py
│   │   └── models.py
│   ├── ml/
│   │   ├── model_loader.py
│   │   ├── predictor.py
│   │   └── preprocessing.py
│   ├── routes/
│   │   ├── auth_routes.py
│   │   ├── dashboard_routes.py
│   │   └── prediction_routes.py
│   ├── services/
│   │   └── history_service.py
│   ├── static/
│   ├── templates/
│   └── main.py
├── data/
├── models/
│   └── ann_scratch_model.pkl
├── notebooks/
├── requirements.txt
└── README.md
```

## Technologies Used

- FastAPI
- Jinja2
- PostgreSQL
- SQLAlchemy
- Passlib bcrypt
- Bootstrap
- HTML/CSS/JavaScript
- NumPy
- Joblib
- ANN model implemented from scratch

## Setup Instructions

1. Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/database_name
SESSION_SECRET_KEY=change-this-development-secret
```

4. Confirm the ANN model file exists:

```text
models/ann_scratch_model.pkl
```

## Database Setup

Create a PostgreSQL database and set `DATABASE_URL` in `.env`.

On application startup, SQLAlchemy creates the known ORM tables if they do not already exist:

- `users`
- `prediction_history`

The startup setup also preserves the existing compatibility helper `ensure_prediction_history_user_id()` for older `prediction_history` tables.

## Run Command

```bash
python3.12 -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## ML Workflow Summary

The prediction flow is intentionally preserved:

1. Read 18 raw browser form values.
2. Validate all values as required, numeric, finite, and non-negative.
3. Apply `log1p` transformation.
4. Compute engineered features:
   - `RainFactor`
   - `LandRisk`
   - `WaterStress`
   - `Blockage`
5. Validate runtime features against the saved model feature list.
6. Build the feature vector using the saved training feature order.
7. Scale values using saved feature means and standard deviations.
8. Run ANN forward propagation with saved parameters.
9. Convert probability to percent.
10. Map risk using the existing thresholds:
    - Low Risk: `< 40`
    - Medium Risk: `< 70`
    - High Risk: `>= 70`
11. Save raw input values and prediction result to `prediction_history`.



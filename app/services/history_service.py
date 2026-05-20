from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import PredictionHistory


def save_prediction_history(
    db: Session,
    raw_values: dict[str, float],
    probability: float,
    risk_level: str,
    user_id: int | None = None,
) -> None:
    """Persist one prediction using raw browser values, not transformed model inputs."""
    history = PredictionHistory(
        **raw_values,
        probability=probability,
        risk_level=risk_level,
        user_id=user_id,
    )
    db.add(history)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def get_user_prediction_summary(
    db: Session,
    user_id: int,
) -> tuple[list[PredictionHistory], int | None, PredictionHistory | None]:
    """Return the latest 20 rows, total count, and newest prediction for a user."""
    history_rows = db.scalars(
        select(PredictionHistory)
        .where(PredictionHistory.user_id == user_id)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(20)
    ).all()
    total_predictions = db.scalar(
        select(func.count())
        .select_from(PredictionHistory)
        .where(PredictionHistory.user_id == user_id)
    )
    latest_prediction = history_rows[0] if history_rows else None

    return history_rows, total_predictions, latest_prediction

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    predictions: Mapped[list["PredictionHistory"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    MonsoonIntensity: Mapped[float] = mapped_column(Float, nullable=False)
    TopographyDrainage: Mapped[float] = mapped_column(Float, nullable=False)
    RiverManagement: Mapped[float] = mapped_column(Float, nullable=False)
    Deforestation: Mapped[float] = mapped_column(Float, nullable=False)
    Urbanization: Mapped[float] = mapped_column(Float, nullable=False)
    ClimateChange: Mapped[float] = mapped_column(Float, nullable=False)
    DamsQuality: Mapped[float] = mapped_column(Float, nullable=False)
    Siltation: Mapped[float] = mapped_column(Float, nullable=False)
    AgriculturalPractices: Mapped[float] = mapped_column(Float, nullable=False)
    Encroachments: Mapped[float] = mapped_column(Float, nullable=False)
    IneffectiveDisasterPreparedness: Mapped[float] = mapped_column(Float, nullable=False)
    DrainageSystems: Mapped[float] = mapped_column(Float, nullable=False)
    Landslides: Mapped[float] = mapped_column(Float, nullable=False)
    Watersheds: Mapped[float] = mapped_column(Float, nullable=False)
    DeterioratingInfrastructure: Mapped[float] = mapped_column(Float, nullable=False)
    PopulationScore: Mapped[float] = mapped_column(Float, nullable=False)
    WetlandLoss: Mapped[float] = mapped_column(Float, nullable=False)
    InadequatePlanning: Mapped[float] = mapped_column(Float, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped[Optional[User]] = relationship(back_populates="predictions")

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.constants import NEPAL_DISTRICTS
from app.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    district: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'user'"))
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
    alert_subscription: Mapped[Optional["AlertSubscription"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )
    alerts_created: Mapped[list["Alert"]] = relationship(back_populates="creator")


class AlertSubscription(Base):
    __tablename__ = "alert_subscriptions"
    __table_args__ = (
        CheckConstraint(
            f"district IN ({', '.join(repr(district) for district in NEPAL_DISTRICTS)})",
            name="ck_alert_subscriptions_district",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False, index=True)
    district: Mapped[str] = mapped_column(String(100), nullable=False)
    email_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="alert_subscription")


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
    district: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    prediction_source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'personal'"),
    )
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped[Optional[User]] = relationship(back_populates="predictions")
    alert: Mapped[Optional["Alert"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_history.id"),
        unique=True,
        nullable=False,
        index=True,
    )
    district: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default=text("'draft'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    prediction: Mapped[PredictionHistory] = relationship(back_populates="alert")
    creator: Mapped[User] = relationship(back_populates="alerts_created")

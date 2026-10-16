from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrbitalRecord(BaseModel):
    """Validated, traceable representation of one public GP/OMM record."""

    model_config = ConfigDict(populate_by_name=True)

    object_id: str = Field(alias="NORAD_CAT_ID")
    object_name: str | None = Field(default=None, alias="OBJECT_NAME")
    epoch: datetime = Field(alias="EPOCH")

    inclination_deg: float = Field(alias="INCLINATION")
    raan_deg: float = Field(alias="RA_OF_ASC_NODE")
    eccentricity: float = Field(alias="ECCENTRICITY")
    arg_pericenter_deg: float = Field(alias="ARG_OF_PERICENTER")
    mean_anomaly_deg: float = Field(alias="MEAN_ANOMALY")
    mean_motion_rev_per_day: float = Field(alias="MEAN_MOTION")

    source: str
    retrieved_at: datetime
    record_format: str = "OMM_JSON"
    quality_flags: list[str] = Field(default_factory=list)

    @field_validator("eccentricity")
    @classmethod
    def eccentricity_must_be_valid(cls, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("ECCENTRICITY must be between 0 and 1.")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def retrieved_time_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
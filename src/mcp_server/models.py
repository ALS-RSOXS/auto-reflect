"""Pydantic response models for MCP tools."""

from pydantic import BaseModel


class AIChannelResponse(BaseModel):
    """Response model for AI channel list."""

    channels: list[str]


class AIValuesResponse(BaseModel):
    """Response model for AI channel values."""

    values: dict[str, float]


class AIUncertaintyResponse(BaseModel):
    """Response model for AI values with uncertainty."""

    values: dict[str, dict[str, float]]


class MotorListResponse(BaseModel):
    """Response model for motor list."""

    motors: list[str]


class MotorPositionsResponse(BaseModel):
    """Response model for motor positions."""

    positions: dict[str, float]


class MotorStatusResponse(BaseModel):
    """Response model for motor status."""

    status: dict[str, dict[str, float | int]]


class DIOChannelsResponse(BaseModel):
    """Response model for DIO channel list."""

    channels: list[str]


class DIOStatesResponse(BaseModel):
    """Response model for DIO channel states."""

    states: dict[str, bool]

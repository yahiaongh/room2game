from typing import Literal

from pydantic import BaseModel


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"


class PlayerStart(StrictModel):
    x: float
    y: float


class ObjectRole(StrictModel):
    object_index: int
    role: Literal["solid"]


class HazardRole(StrictModel):
    object_index: int
    role: Literal["hazard"]
    damage: int


class CollectibleRole(StrictModel):
    object_index: int
    role: Literal["collectible"]
    points: int


class GameRules(StrictModel):
    player_start: PlayerStart
    platforms: list[ObjectRole]
    hazards: list[HazardRole]
    collectibles: list[CollectibleRole]
    win_condition: str
    theme: str


class RestyleRequest(StrictModel):
    image_id: str
    object_index: int
    theme: str

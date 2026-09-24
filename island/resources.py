"""Food production, storage, consumption, and shortage calculations."""
from dataclasses import dataclass
import math

from .config import FoodConfig


@dataclass(frozen=True)
class FoodResult:
    produced: float
    spoiled: float
    consumed: float
    unmet: float
    ending_stock: float
    security: float


def update_food(stock: float, storage_limit: float, land_yield: float, land_health: float,
                adults: int, children: int, elders: int, cfg: FoodConfig) -> FoodResult:
    labor_factor = 1.0 - math.exp(-max(0, adults) / cfg.labor_saturation)
    produced = max(0.0, land_yield * land_health * labor_factor * cfg.adult_labor_productivity)
    available = max(0.0, stock) + produced
    biological_spoilage = available * cfg.spoilage_fraction
    after_spoilage = available - biological_spoilage
    overflow = max(0.0, after_spoilage - max(0.0, storage_limit))
    spoiled = biological_spoilage + overflow
    available = after_spoilage - overflow
    need = (children * cfg.child_consumption + adults * cfg.adult_consumption
            + elders * cfg.elder_consumption)
    consumed = min(available, need)
    unmet = max(0.0, need - consumed)
    security = 1.0 if need <= 0 else consumed / need
    return FoodResult(produced, spoiled, consumed, unmet, available - consumed, security)


def smoothed_security(previous: float, current: float, cfg: FoodConfig) -> float:
    memory = min(max(cfg.security_memory, 0.0), 1.0)
    return memory * previous + (1.0 - memory) * current
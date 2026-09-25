"""A strategy is a parameter record, never code — every variant is testable and reversible."""
import hashlib
import json
from dataclasses import dataclass, asdict, fields


@dataclass(frozen=True)
class Strategy:
    type: str
    delta_target: float
    dte: int
    ivr_min: float = 0.0
    trend_filter: bool = False
    take_profit: float | None = 0.5
    roll_at_dte: int | None = 21
    close_before_earnings: bool = True
    hold_to_expiry: bool = False

    def __post_init__(self):
        errs = []
        if self.type not in ("CC", "CSP"):
            errs.append("type must be CC or CSP")
        if not 0.15 <= self.delta_target <= 0.35:
            errs.append("delta_target must be within 0.15-0.35")
        if not 21 <= self.dte <= 45:
            errs.append("dte must be within 21-45")
        if not 0 <= self.ivr_min <= 100:
            errs.append("ivr_min must be within 0-100")
        if self.take_profit is not None and not 0 < self.take_profit < 1:
            errs.append("take_profit must be in (0,1) or None")
        if self.roll_at_dte is not None and not 0 < self.roll_at_dte < self.dte:
            errs.append("roll_at_dte must be in (0, dte) or None")
        if self.hold_to_expiry and self.roll_at_dte is not None:
            errs.append("hold_to_expiry requires roll_at_dte=None")
        if errs:
            raise ValueError("; ".join(errs))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def id(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown keys: {sorted(unknown)}")
        return cls(**d)

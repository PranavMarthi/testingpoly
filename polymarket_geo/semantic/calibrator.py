from __future__ import annotations

from dataclasses import dataclass

from polymarket_geo.semantic.embedder import sigmoid


@dataclass(frozen=True)
class LinearCalibrationParams:
    bias: float = -2.4
    w_combined: float = 3.0
    w_title: float = 1.8
    w_desc: float = 1.4
    w_choices: float = 1.2
    w_agreement: float = 0.8
    w_importance: float = 0.6


class Calibrator:
    """Small calibrated linear model for confidence estimation."""

    def __init__(self, params: LinearCalibrationParams | None = None):
        self.params = params or LinearCalibrationParams()

    def confidence(
        self,
        *,
        s_combined: float,
        s_title: float,
        s_desc: float,
        s_choices: float,
        agreement: float,
        importance: float,
    ) -> float:
        p = self.params
        z = (
            p.bias
            + p.w_combined * s_combined
            + p.w_title * s_title
            + p.w_desc * s_desc
            + p.w_choices * s_choices
            + p.w_agreement * agreement
            + p.w_importance * importance
        )
        return max(0.0, min(1.0, sigmoid(z)))

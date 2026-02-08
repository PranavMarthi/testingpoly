from __future__ import annotations


class GeoTypeDecider:
    def decide(self, confidences: list[float], event_type: str) -> str:
        if not confidences:
            return "none"

        top = confidences[0]
        second = confidences[1] if len(confidences) > 1 else 0.0
        margin = top - second

        if top < 0.09:
            return "none"
        if top < 0.2 and margin < 0.04:
            return "ambiguous"
        if len(confidences) > 1 and top >= 0.2 and margin < 0.06:
            return "multi"
        if event_type == "global" and top < 0.2:
            return "global"
        return "inferred"

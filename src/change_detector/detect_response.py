from dataclasses import dataclass
from src.change_detector.control_state import ControlState

@dataclass
class DetectResponse:
    state: ControlState
    alpha_estimate: float|None
    beta_estimate: float|None
    min_u1: float|None
    min_u2: float|None
    min_u3: float|None
    current_u1: float|None
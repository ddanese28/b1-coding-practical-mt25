from __future__ import annotations
from typing import Optional
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class PDController:
    """Discrete PD controller.

    Implements: u[t] = kp * e[t] + kd * (e[t] - e[t-1])

    Default gains match the spec: kp=0.15, kd=0.6.
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6, u_min: Optional[float] = None, u_max: Optional[float] = None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.u_min = None if u_min is None else float(u_min)
        self.u_max = None if u_max is None else float(u_max)
        self._prev_error: Optional[float] = None

    def compute(self, error: float, prev_error: Optional[float] = None) -> float:
        if error is None or (isinstance(error, float) and np.isnan(error)):
            raise ValueError("error must be numeric")

        if prev_error is None:
            prev = 0.0 if self._prev_error is None else float(self._prev_error)
        else:
            prev = float(prev_error)

        derivative = float(error) - float(prev)
        u = self.kp * float(error) + self.kd * derivative

        # save for next call
        self._prev_error = float(error)

        # clamp
        if (self.u_min is not None) and (u < self.u_min):
            u = self.u_min
        if (self.u_max is not None) and (u > self.u_max):
            u = self.u_max

        return float(u)

    def step(self, r: float, y: float) -> float:
        if r is None or y is None:
            raise ValueError("r and y must be numeric")
        e = float(r) - float(y)
        return self.compute(e)

    def __call__(self, error: float, prev_error: Optional[float] = None) -> float:
        return self.compute(error, prev_error)


if __name__ == "__main__":
    c = PDController()
    print("PD test u:", c.step(1.0, 0.0))
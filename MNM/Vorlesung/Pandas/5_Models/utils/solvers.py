from typing import Generator, Optional, Tuple

import numpy as np
from scipy.optimize import newton

#  typedefs
fTupleGenerator = Generator[Tuple[float, float], None, None]


def euler_explicit(
    t_span: tuple, t_step: float, v_init: float, rhs: callable
) -> fTupleGenerator:
    t_min, t_max = t_span
    # h = (max - min) / (N + 1 - 1)
    N: int = int((t_max - t_min) / t_step)

    v_current: float = v_init
    t_current: float = t_min
    for it in range(N + 1):
        yield t_current, v_current
        t_current += t_step
        v_current += rhs(v_current) * t_step


def euler_implicit(
    t_span: tuple, t_step: float, v_init: float, rhs: callable
) -> fTupleGenerator:
    t_min, t_max = t_span
    # h = (max - min) / (N + 1 - 1)
    N: int = int((t_max - t_min) / t_step)

    v_current: float = v_init
    t_current: float = t_min
    for it in range(N + 1):
        yield t_current, v_current
        t_current += t_step
        v_current = newton(
            func=lambda v: v - (v_current + t_step * rhs(v)), x0=v_current
        )


class Solution:
    def __init__(self, sol_generator: fTupleGenerator) -> None:
        self.sol_generator: fTupleGenerator = sol_generator

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        xval, yval = [], []
        for (x, y) in self.sol_generator:
            xval.append(x)
            yval.append(y)
        return (np.asarray(xval), np.asarray(yval))


class EulerSolver:
    def __init__(
        self, x_span: Tuple[float], x_step: float, x_init: float, rhs: callable
    ) -> None:
        self.x_span: Tuple[float] = x_span
        self.x_step: float = x_step
        self.x_init: float = x_init
        self.rhs: callable = rhs
        self.x_values: Optional[np.ndarray] = None
        self.y_values: Optional[np.ndarray] = None
        self.methods = {
            'euler-explicit': euler_explicit,
            'euler-implicit': euler_implicit,
        }

    def solve(self, method: str = '') -> Solution:
        if not method:
            raise ValueError('Most provide solution method.')
        solver = self.methods.get(method)
        if solver is not None:
            sol = solver(self.x_span, self.x_step, self.x_init, self.rhs)
            return Solution(sol)
        else:
            raise ValueError('This is not a valid solver.')

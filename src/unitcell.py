"""
Created on Fri Jun 20 2025

@author: MMCS-CSY

This module defines the UnitCell class for generating and manipulating
surface lattice structures under periodic boundary conditions (PBC).
It leverages NumPy for array operations and Numba for JIT acceleration.
"""
import numpy as np
from numba import njit, prange
from typing import Optional, Literal, List
from dataclasses import dataclass

@dataclass
class UnitCell:
    """
    Represents a unit cell of surface structures with periodic boundaries.

    Attributes
    ----------
    width : float
        Width of structures on the surface (micrometers).
    spacing : float
        Spacing between structures on the surface (micrometers).
    height : float
        Height of the surface structures (micrometers).
    shape : Literal['rec', 'cyl']
        Geometry of the structures: 'rec' for rectangular, 'cyl' for cylindrical.
    ca_int : float
        Contact angle on a flat surface without structures (degrees).
    ca_exp : float
        Expected contact angle on the structured surface (degrees).
    grid : int
        Default grid resolution for lattice generation.
    """
    width: Optional[float] = None
    spacing: Optional[float] = None
    height: Optional[float] = None
    shape: Optional[Literal["rec", "cyl"]] = None
    ca_int: Optional[float] = None
    ca_exp: Optional[float] = None
    grid: int = 10
    dL: Optional[float] = None  # 자동계산할 필드


    def __post_init__(self):
        if self.width is not None and self.spacing is not None and self.grid > 1:
            self.dL = (self.width + self.spacing) / (self.grid - 1)

 
    def __repr__(self) -> str:
        """
        Return a string representation of the UnitCell.
        """
        return (
            f"UnitCell(width={self.width} um, spacing={self.spacing} um, "
            f"height={self.height} um, shape='{self.shape}', "
            f"ca_int={self.ca_int} deg, ca_exp={self.ca_exp} deg, grid={self.grid})"
        )

    def lattice(self, grid: Optional[int] = None) -> np.ndarray:
        """
        Generate the base lattice array for the unit cell.

        Parameters
        ----------
        grid : int, optional
            Number of grid points per axis (default=self.grid).

        Returns
        -------
        np.ndarray
            2D array of shape (grid, grid) representing height map of structures.
        """
        use_grid = self.grid if grid is None else grid
        return generate_lattice(
            float(self.width),
            float(self.spacing),
            float(self.height),
            self.shape,
            use_grid,
        )

    def lattice_pbc(
        self,
        direction: Literal['H', 'V', 'D'],
        grid: Optional[int] = None,
        pbc_step: int = 1,
    ) -> List[np.ndarray]:
        """
        Apply periodic boundary shifts to the base lattice.

        Parameters
        ----------
        direction : {'H', 'V', 'D'}
            Shift direction: 'H' horizontal, 'V' vertical, 'D' diagonal.
        grid : int, optional
            Number of grid points per axis (default=self.grid).
        pbc_step : int, optional
            Step interval for PBC shifts (default=1).

        Returns
        -------
        List[np.ndarray]
            List of shifted lattices for each PBC step.
        """
        use_grid = self.grid if grid is None else grid
        base_lattice = self.lattice(use_grid)
        return pbc_moving(base_lattice, direction, use_grid, pbc_step)

@njit(parallel=True, fastmath=True)
def generate_lattice(
    width: float,
    spacing: float,
    height: float,
    shape: str,
    grid: int,
) -> np.ndarray:
    """
    Generate a height map for the unit cell based on geometry parameters.

    Parameters
    ----------
    width : float
        Width of structures (micrometers).
    spacing : float
        Spacing between structures (micrometers).
    height : float
        Height to apply where structures exist (micrometers).
    shape : {'rec', 'cyl'}
        Geometry type: rectangular or cylindrical.
    grid : int
        Number of grid points per axis.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (grid, grid), where each entry is either 0
        (no structure) or 'height' (structure present).
    """
    w, s, h = width, spacing, height
    L = w + s
    dL = L / (grid - 1)
    lattice = np.ones((grid, grid), dtype=np.float32)

    if shape == 'rec':
        for xi in prange(grid):
            x_temp = xi * dL
            for yi in range(grid):
                y_temp = yi * dL
                if ((x_temp > 0.5 * w) and (x_temp < 0.5 * w + s)) or (
                    (y_temp > 0.5 * w) and (y_temp < 0.5 * w + s)
                ):
                    lattice[xi, yi] = 0.0
                else:
                    lattice[xi, yi] = h
    elif shape == 'cyl':
        r = 0.5 * w
        r_sq = r * r
        half_w = 0.5 * w
        half_w_plus_s = half_w + s
        for xi in prange(grid):
            x_temp = xi * dL
            for yi in range(grid):
                y_temp = yi * dL
                cond1 = (
                    (0 <= x_temp <= half_w)
                    and (0 <= y_temp <= half_w)
                    and (x_temp * x_temp + y_temp * y_temp <= r_sq)
                )
                cond2 = (
                    (0 <= x_temp <= half_w)
                    and (half_w_plus_s <= y_temp <= L)
                    and (x_temp * x_temp + (y_temp - L) * (y_temp - L) <= r_sq)
                )
                cond3 = (
                    (half_w_plus_s <= x_temp <= L)
                    and (0 <= y_temp <= half_w)
                    and ((x_temp - L) * (x_temp - L) + y_temp * y_temp <= r_sq)
                )
                cond4 = (
                    (half_w_plus_s <= x_temp <= L)
                    and (half_w_plus_s <= y_temp <= L)
                    and ((x_temp - L) * (x_temp - L) + (y_temp - L) * (y_temp - L) <= r_sq)
                )
                if cond1 or cond2 or cond3 or cond4:
                    lattice[xi, yi] = h
                else:
                    lattice[xi, yi] = 0.0
    else:
        for xi in prange(grid):
            for yi in range(grid):
                lattice[xi, yi] = 0.0
    return lattice


def pbc_moving(
    lattice: np.ndarray,
    direction: Literal['H', 'V', 'D'],
    grid: int,
    pbc_step: int,
) -> List[np.ndarray]:
    """
    Apply periodic boundary condition shifts to a lattice.

    Parameters
    ----------
    lattice : np.ndarray
        2D array of the base lattice.
    direction : {'H', 'V', 'D'}
        Shift direction: H-horizontal, V-vertical, D-diagonal.
    grid : int
        Number of points per axis of the lattice.
    pbc_step : int
        Interval of shift steps to apply.

    Returns
    -------
    List[np.ndarray]
        List of shifted lattices for each PBC multiple of pbc_step.
    """
    n_steps = [step for step in range(1, grid) if step % pbc_step == 0]
    lattice_pbc = []
    for shift in n_steps:
        if direction == 'H':
            shifted = np.roll(lattice, shift, axis=1)
        elif direction == 'V':
            shifted = np.roll(lattice, shift, axis=0)
        else:
            shifted = np.roll(np.roll(lattice, shift, axis=0), shift, axis=1)
        lattice_pbc.append(shifted)
    return lattice_pbc

if __name__ == "__main__":
    unit_cell = UnitCell(width=10, spacing=5, height=2, shape='rec', ca_int=90, ca_exp=120)
    print(unit_cell)
    print(unit_cell.dL)
    lattice = unit_cell.lattice()
    print(lattice)
    pbc_lattices = unit_cell.lattice_pbc(direction='D', pbc_step=1)
    for idx, shifted in enumerate(pbc_lattices):
        print(f"Shift {idx+1}:")
        print(shifted)

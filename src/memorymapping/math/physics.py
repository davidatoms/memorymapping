from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple
from .landscape import NeuralLandscape

@dataclass
class Particle:
    """A particle moving on the energy landscape"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    trajectory_x: List[float] = field(default_factory=list)
    trajectory_y: List[float] = field(default_factory=list)
    energy_history: List[Tuple[float, float, float]] = field(default_factory=list) # (KE, PE, Total)

    def get_kinetic_energy(self) -> float:
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)
        
    def get_potential_energy(self, landscape: NeuralLandscape) -> float:
        # We assume landscape generate_landscape works on scalars too or use a helper
        # Since generate_landscape expects arrays, we simply implement the formula logic or call it with 1x1
        # For efficiency, let's just re-calculate the Potential V at (x,y)
        # However, to avoid duplication, we should probably add a V(x,y) method to NeuralLandscape.
        # For now, I'll inline the formula to match generate_landscape
        
        r_sq = self.x**2 + self.y**2
        valley = -3.0 * np.exp(-r_sq / 8.0)
        
        ridges = 0.0
        ridge_centers = [(3, 3), (-3, -3), (3, -3), (-3, 3)]
        for cx, cy in ridge_centers:
            d_sq = (self.x - cx)**2 + (self.y - cy)**2
            ridges += 2.0 * np.exp(-d_sq / 2.0)
            
        slope = 0.3 * (self.x + self.y)
        
        return valley + ridges + slope

class HamiltonianSystem:
    """Manages the simulation of particles using Hamiltonian dynamics"""
    
    def __init__(self, landscape: NeuralLandscape):
        self.landscape = landscape
        self.particles: List[Particle] = []
        
    def add_particle(self, x, y, vx=0, vy=0, mass=1.0):
        p = Particle(x, y, vx, vy, mass)
        self.particles.append(p)
        
    def step(self, dt: float = 0.01):
        """
        Symplectic Euler Integration Step
        1. v(t+1) = v(t) + F(x(t)) * dt / m
        2. x(t+1) = x(t) + v(t+1) * dt
        """
        for p in self.particles:
            # 1. Compute Force F = -grad(V)
            dv_dx, dv_dy = self.landscape.gradient(p.x, p.y)
            fx = -dv_dx
            fy = -dv_dy
            
            # Friction/Damping (Optional, keeps it stable else energy conservation error accumulates)
            # Pure Hamiltonian = No friction.
            # Let's add very slight damping to mimic 'entropy loss'? Or keep pure for "Energy Spikes" demo.
            # User wants "Energy Spikes", usually implies conservation exchange.
            
            # 2. Update Velocity
            ax = fx / p.mass
            ay = fy / p.mass
            
            p.vx += ax * dt
            p.vy += ay * dt
            
            # 3. Update Position
            p.x += p.vx * dt
            p.y += p.vy * dt
            
            # Record State
            p.trajectory_x.append(p.x)
            p.trajectory_y.append(p.y)
            
            ke = p.get_kinetic_energy()
            pe = p.get_potential_energy(self.landscape)
            p.energy_history.append((ke, pe, ke + pe))

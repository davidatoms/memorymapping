import numpy as np

class NeuralLandscape:
    """
    Defines the energy landscape for memory states.
    Includes both generation of the heightmap and analytical gradients for physics.
    """
    
    @staticmethod
    def generate_landscape(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """
        Generates the Z values (Energy/Entropy) for the given grid.
        Formula:
        Z = -3 * exp(-r^2/8)  (Central Valley)
          + sum(2 * exp(-((x-cx)^2 + (y-cy)^2)/2)) (Ridges)
          + 0.3 * (x+y) (General Slope/Bias)
        """
        # 1. Central Deep Valley (Attractor)
        r_sq = x_grid**2 + y_grid**2
        valley = -3.0 * np.exp(-r_sq / 8.0)
        
        # 2. Ridges (Barriers)
        ridges = np.zeros_like(x_grid)
        ridge_centers = [(3, 3), (-3, -3), (3, -3), (-3, 3)]
        for cx, cy in ridge_centers:
            d_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            ridges += 2.0 * np.exp(-d_sq / 2.0)
            
        # 3. Overall Slope (Drift)
        slope = 0.3 * (x_grid + y_grid)
        
        return valley + ridges + slope

    def gradient(self, x: float, y: float) -> tuple[float, float]:
        """
        Calculate analytical gradient (dV/dx, dV/dy) at position (x,y).
        Used for physics force calculation: F = -grad(V)
        """
        # 1. Central Valley: Vc = -3 * exp(-(x^2+y^2)/8)
        # dVc/dx = -3 * exp(...) * (-2x/8) = 0.75 * x * exp(...)
        r_sq = x**2 + y**2
        exp_valley = np.exp(-r_sq / 8.0)
        dVc_dx = 0.75 * x * exp_valley
        dVc_dy = 0.75 * y * exp_valley
        
        # 2. Ridges: Vr = 2 * exp(-((x-cx)^2 + (y-cy)^2)/2)
        # dVr/dx = 2 * exp(...) * (-2(x-cx)/2) = -2(x-cx) * exp(...)
        ridge_grad_x = 0.0
        ridge_grad_y = 0.0
        ridge_centers = [(3, 3), (-3, -3), (3, -3), (-3, 3)]
        
        for cx, cy in ridge_centers:
            d_sq = (x - cx)**2 + (y - cy)**2
            exp_ridge = np.exp(-d_sq / 2.0)
            ridge_grad_x += -2.0 * (x - cx) * exp_ridge
            ridge_grad_y += -2.0 * (y - cy) * exp_ridge
            
        # 3. Slope: Vs = 0.3(x+y) -> dVs/dx = 0.3
        slope_grad_x = 0.3
        slope_grad_y = 0.3
        
        total_grad_x = dVc_dx + ridge_grad_x + slope_grad_x
        total_grad_y = dVc_dy + ridge_grad_y + slope_grad_y
        
        return total_grad_x, total_grad_y

def generate_valley_landscape(n=100, limit=5):
    """Legacy wrapper for compatibility"""
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    X, Y = np.meshgrid(x, y)
    Z = NeuralLandscape.generate_landscape(X, Y)
    return X, Y, Z

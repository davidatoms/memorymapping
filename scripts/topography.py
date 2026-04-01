import matplotlib.pyplot as plt
import numpy as np

# Create coordinate grids
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)

# Create figure with subplots for different shapes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Simple Topography Shapes', fontsize=16)

# 1. Mountain (cone)
Z1 = 10 - np.sqrt(X**2 + Y**2)
Z1[Z1 < 0] = 0
axes[0,0].contourf(X, Y, Z1, levels=15, cmap='terrain')
axes[0,0].set_title('Mountain (Cone)')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')

# 2. Valley (inverted cone)
Z2 = np.sqrt(X**2 + Y**2) - 5
Z2[Z2 < 0] = 0
axes[0,1].contourf(X, Y, Z2, levels=15, cmap='terrain')
axes[0,1].set_title('Valley')
axes[0,1].set_xlabel('X')
axes[0,1].set_ylabel('Y')

# 3. Ridge (linear slope)
Z3 = 10 - np.abs(Y)
axes[0,2].contourf(X, Y, Z3, levels=15, cmap='terrain')
axes[0,2].set_title('Ridge')
axes[0,2].set_xlabel('X')
axes[0,2].set_ylabel('Y')

# 4. Saddle point
Z4 = X**2 - Y**2
axes[1,0].contourf(X, Y, Z4, levels=15, cmap='terrain')
axes[1,0].set_title('Saddle Point')
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Y')

# 5. Multiple peaks
Z5 = 8*np.exp(-(X-3)**2 - (Y-3)**2) + 6*np.exp(-(X+3)**2 - (Y+3)**2) + 4*np.exp(-(X+3)**2 - (Y-3)**2)
axes[1,1].contourf(X, Y, Z5, levels=15, cmap='terrain')
axes[1,1].set_title('Multiple Peaks')
axes[1,1].set_xlabel('X')
axes[1,1].set_ylabel('Y')

# 6. Plateau
Z6 = np.where((np.abs(X) < 5) & (np.abs(Y) < 5), 8, 2)
axes[1,2].contourf(X, Y, Z6, levels=15, cmap='terrain')
axes[1,2].set_title('Plateau')
axes[1,2].set_xlabel('X')
axes[1,2].set_ylabel('Y')

plt.tight_layout()
plt.show()

# Alternative: Single shape with contour lines
plt.figure(figsize=(10, 8))

# Create a more complex mountain range
Z_complex = (8*np.exp(-(X-2)**2 - (Y-2)**2) + 
             6*np.exp(-(X+3)**2 - (Y+1)**2) + 
             4*np.exp(-(X+1)**2 - (Y-4)**2) + 
             3*np.exp(-(X-4)**2 - (Y+3)**2))

# Plot with filled contours
plt.contourf(X, Y, Z_complex, levels=20, cmap='terrain', alpha=0.8)
plt.colorbar(label='Elevation')

# Add contour lines
plt.contour(X, Y, Z_complex, levels=10, colors='black', alpha=0.5, linewidths=0.5)

plt.title('Mountain Range with Contour Lines')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, alpha=0.3)
plt.show()

# Simple 3D version
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# 3D Mountain
ax1 = fig.add_subplot(121, projection='3d')
Z_mountain = 10 - np.sqrt(X**2 + Y**2)
Z_mountain[Z_mountain < 0] = 0
ax1.plot_surface(X, Y, Z_mountain, cmap='terrain', alpha=0.8)
ax1.set_title('3D Mountain')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Elevation')

# 3D Saddle
ax2 = fig.add_subplot(122, projection='3d')
Z_saddle = X**2 - Y**2
ax2.plot_surface(X, Y, Z_saddle, cmap='terrain', alpha=0.8)
ax2.set_title('3D Saddle')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Elevation')

plt.tight_layout()
plt.show()
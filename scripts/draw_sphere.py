import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import datetime

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere surface
ax.plot_surface(x, y, z, color='b', alpha=0.1)
# Plot wireframe for better visibility
ax.plot_wireframe(x, y, z, color='b', alpha=0.2, linewidth=0.5)

# Plot origin
ax.scatter([0], [0], [0], color='black', s=50, label='Origin')

# Customize the plot
ax.set_title('Unit Sphere')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.legend()

# Save the plot first
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("graphs", exist_ok=True)
save_path = os.path.join("graphs", f"sphere_{timestamp}.png")
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Plot saved to {save_path}")

# Show the plot
plt.show()

# Close the plot to free memory
plt.close() 
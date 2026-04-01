import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Create figure
plt.figure(figsize=(8, 8))
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Draw unit circle
circle = plt.Circle((0, 0), 1, fill=False, color='blue', linestyle='--', label='Unit Circle')
plt.gca().add_patch(circle)

# Generate random point
random_x = np.random.uniform(-1, 1)
random_y = np.random.uniform(-1, 1)

# Calculate distance from origin
distance_from_origin = np.sqrt(random_x**2 + random_y**2)

# Plot the random point
plt.plot(random_x, random_y, 'ro', markersize=10, 
         label=f'Random Point ({random_x:.3f}, {random_y:.3f})\nDistance from origin: {distance_from_origin:.3f}')

# Plot a line from origin to the point
plt.plot([0, random_x], [0, random_y], 'r--', alpha=0.5)

# Add origin point
plt.plot(0, 0, 'ko', markersize=5, label='Origin')

# Customize the plot
plt.title('Random Point with Unit Circle')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Make sure circle appears circular

# Show the plot
plt.show()

# Save the plot
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("graphs", exist_ok=True)
save_path = os.path.join("graphs", f"random_point_{timestamp}.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
print(f"Random point coordinates: ({random_x:.3f}, {random_y:.3f})")
print(f"Distance from origin: {distance_from_origin:.3f}")
print(f"Point is {'inside' if distance_from_origin <= 1 else 'outside'} the unit circle") 
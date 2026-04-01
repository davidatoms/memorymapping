import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# Create figure
fig = plt.figure(figsize=(8, 8))
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Draw unit circle
circle = plt.Circle((0, 0), 1, fill=False, color='blue', linestyle='--', label='Unit Circle')
plt.gca().add_patch(circle)

# Generate random points inside the circle using rejection sampling
num_points = 50  # Number of points to generate
points_x = []
points_y = []
count = 0

while count < num_points:
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    # Check if point is inside circle (x² + y² ≤ 1)
    if x*x + y*y <= 1:
        points_x.append(x)
        points_y.append(y)
        count += 1

# Plot the random points
plt.scatter(points_x, points_y, c='red', s=30, alpha=0.6, label=f'{num_points} Random Points')

# Add origin point
plt.plot(0, 0, 'ko', markersize=5, label='Origin')

# Customize the plot
plt.title('Random Points Inside Unit Circle')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Make sure circle appears circular

# Save the plot first
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("graphs", exist_ok=True)
save_path = os.path.join("graphs", f"random_points_circle_{timestamp}.png")
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Plot saved to {save_path}")

# Show the plot
plt.show()

# Close the plot to free memory
plt.close() 
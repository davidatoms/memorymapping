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

# Add origin point
plt.plot(0, 0, 'ko', markersize=5, label='Origin')

# Customize the plot
plt.title('Unit Circle')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Make sure circle appears circular

# Save the plot first
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("graphs", exist_ok=True)
save_path = os.path.join("graphs", f"circle_{timestamp}.png")
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"Plot saved to {save_path}")

# Show the plot
plt.show()

# Close the plot to free memory
plt.close() 
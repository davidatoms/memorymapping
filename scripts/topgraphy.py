import numpy as np
import matplotlib.pyplot as plt

# 1. Generate or load elevation data
# For demonstration, we'll create a simple 2D array representing elevation
# In a real-world scenario, you would load this from a file (e.g., GeoTIFF)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.5 * np.cos(X) # Example elevation data

# 2. Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=10, cmap='terrain') # levels controls number of contour lines
plt.colorbar(contour, label='Elevation') # Add a colorbar for reference

# 3. Customize the plot (optional)
plt.title('Topographic Map')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.show()

# 4. Save the plot with date and hash
import datetime
import hashlib
import os

# Generate timestamp and hash
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
data_hash = hashlib.md5(str(Z.tobytes()).encode()).hexdigest()[:8]

save_plot = input("Do you want to save the plot? (y/n): ")
if save_plot == "y":
    file_name = input("Enter the file name: ")
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"/{file_name}_{timestamp}_{data_hash}.png")
    print(f"Plot saved to {file_name}")
else:
    print("Plot not saved")
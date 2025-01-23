import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax - xmin) * pixel_density))
    im = np.linspace(ymin, ymax, int((ymax - ymin) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


def is_stable(c, num_iterations):
    z = 0
    for _ in range(num_iterations):
        z = z**2 + c
    return abs(z) <= 2


def get_members(c, num_iterations):
    mask = is_stable(c, num_iterations)
    return c[mask]


# Generate Mandelbrot set
c = complex_matrix(-1.5, 0.5, -1, 1, pixel_density=1024)

# Create figure and save initial PNG
plt.figure(figsize=(10, 10))
plt.imshow(is_stable(c, num_iterations=20), cmap="binary")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig("./static/apple-touch-icon.png", transparent=True, dpi=300)
plt.close()

# Image sizes to export (in pixels)
sizes = [16, 32, 192, 512]
names = ["favicon", "favicon", "android-chrome", "android-chrome"]

# Open the original image
img = Image.open("./static/apple-touch-icon.png")

# Export PNGs in different sizes
for size, name in zip(sizes, names):
    resized_img = img.resize((size, size), Image.LANCZOS)
    resized_img.save(f"./static/{name}-{size}x{size}.png")

# Create ICO file with multiple sizes
icon_sizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
img.save("./static/favicon.ico", sizes=icon_sizes)

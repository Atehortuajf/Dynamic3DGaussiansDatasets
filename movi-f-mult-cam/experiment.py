import math

# Function to generate equidistant points on a hemisphere (taken from Touch3DGS)
def fibonacci_hemisphere(samples, sphere_radius):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        z = (i / float(samples - 1))  # Range from 0 to 1
        radius = math.sqrt(1 - z * z)  # Radius at y

        theta = phi * i  # Increment

        x = math.cos(theta) * radius * sphere_radius
        y = math.sin(theta) * radius * sphere_radius

        points.append((x, y, z * sphere_radius))

    return points
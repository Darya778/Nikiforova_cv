import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
import numpy as np


def count_colors(fig_colors):
    unique_colors = set(fig_colors)
    for color in unique_colors:
        occurrences = fig_colors.count(color)
        print(f"{color}: {occurrences}")


def hue_to_color_name(hue_value):
    if hue_value < 0.05 or hue_value >= 0.95:
        return "Red"
    if 0.05 <= hue_value < 0.15:
        return "Orange"
    if 0.15 <= hue_value < 0.35:
        return "Yellow"
    if 0.35 <= hue_value < 0.45:
        return "Green"
    if 0.45 <= hue_value < 0.65:
        return "Cyan"
    if 0.65 <= hue_value < 0.75:
        return "Blue"
    if 0.75 <= hue_value < 0.85:
        return "Purple"
    if 0.85 <= hue_value < 0.95:
        return "Pink"
    return "Unknown"


image = plt.imread("balls_and_rects.png")
image_hsv = rgb2hsv(image)

gray_image = image.mean(axis=2)
binary_mask = np.where(gray_image > 0, 1, 0)
labeled_image = label(binary_mask)
detected_regions = regionprops(labeled_image)

shape_data = {"circles": 0, "rectangles": 0}
circle_colors = []
rectangle_colors = []

for region in detected_regions:
    shape_type = "circles" if region.eccentricity < 0.5 else "rectangles"
    shape_data[shape_type] += 1

    centroid_y, centroid_x = region.centroid
    hue = image_hsv[int(centroid_y), int(centroid_x)][0]
    color_name = hue_to_color_name(hue)

    if shape_type == "circles":
        circle_colors.append(color_name)
    else:
        rectangle_colors.append(color_name)

# Display results
print(f"Total shapes: {np.max(labeled_image)}")
print(f"Circles: {shape_data['circles']}")
print(f"Rectangles: {shape_data['rectangles']}\n")
print("Circle Colors:")
count_colors(circle_colors)
print("\nRectangle Colors:")
count_colors(rectangle_colors)

import cv2  # pip install opencv.python
import numpy as np
import os


def compute_statistics(shapes, area_threshold=1000, ratio_threshold=2):
    shape_areas = []
    shape_ratios = []

    for s in shapes:
        if len(s) > 2:
            area = cv2.contourArea(s)
            if area > area_threshold:
                bounding_box = cv2.minAreaRect(s)
                w, h = bounding_box[1]
                ratio = max(w, h) / min(w, h)

                if ratio >= ratio_threshold:
                    shape_areas.append(area)
                    shape_ratios.append(ratio)

    median_area = np.median(shape_areas) if shape_areas else 0
    median_ratio = np.median(shape_ratios) if shape_ratios else 0

    return median_area, median_ratio


def detect_objects(shapes, area_min, ratio_min):
    object_count = 0
    for s in shapes:
        if len(s) > 2:
            bounding_box = cv2.minAreaRect(s)
            area = cv2.contourArea(s)

            w, h = bounding_box[1]
            ratio = max(w, h) / min(w, h)

            if area >= area_min and ratio >= ratio_min:
                object_count += 1
    return object_count


color_ranges = {'orange': ((5, 100, 100), (20, 255, 255)),
                'green': ((30, 50, 50), (85, 255, 255)),
                'blue': ((90, 50, 50), (140, 255, 255))}


total_objects = 0
all_shapes = []

for i in range(1, 13):
    img_path = f'images/img ({i}).jpg'

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading file {img_path}")
        continue

    blurred_img = cv2.GaussianBlur(image, (11, 11), 0)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in color_ranges.items():
        color_mask = cv2.inRange(hsv_img, lower, upper)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, None)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_shapes.extend(contours)

area_min, ratio_min = compute_statistics(all_shapes, 5000, 2)

print(f"Area min: {area_min}, Ratio min: {ratio_min}")

for i in range(1, 13):
    img_path = f'images/img ({i}).jpg'
    print(f"Searching in image {i}")

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading file {img_path}")
        continue

    blurred_img = cv2.GaussianBlur(image, (11, 11), 0)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in color_ranges.items():
        color_mask = cv2.inRange(hsv_img, lower, upper)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, None)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_objects = detect_objects(contours, area_min, ratio_min)
        total_objects += found_objects
        print(f"{color}: {found_objects}")

print(f"Total pencils: {total_objects}")

import cv2
import numpy as np

image_path = "screenshot.png"
image = cv2.imread(image_path)
total_count = 0


def count_objects(image_input):
    circles_count = 0
    squares_count = 0
    total_count = 0
    hsv_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    brightness_channel = hsv_image[:, :, 2]

    ret, threshold_saturation = cv2.threshold(saturation_channel, 50, 255, cv2.THRESH_BINARY)
    ret, threshold_brightness = cv2.threshold(brightness_channel, 225, 255, cv2.THRESH_BINARY)

    combined_threshold = cv2.bitwise_or(threshold_saturation, threshold_brightness)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(combined_threshold, kernel, iterations=2)

    kernel = np.ones((3, 3), np.uint8)
    morphed_image = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Binary", morphed_image)

    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(approximated_contour) > 6:
            circles_count += 1
        total_count += 1

    squares_count = total_count - circles_count
    return circles_count, squares_count


while True:
    current_frame = image
    object_counts = count_objects(current_frame)
    key = cv2.waitKey(100)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    if key == ord('s'):
        cv2.imwrite("image.jpg", current_frame)

    cv2.putText(current_frame, f"Circle - {object_counts[0]}, Square - {object_counts[1]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    count_objects(current_frame)
    cv2.imshow("Client Recv", current_frame)

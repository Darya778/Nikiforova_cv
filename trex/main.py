import time
import cv2
import numpy as np
import pyautogui
import mss

SCREEN_REGION = {'top': 280, 'left': 600, 'width': 700, 'height': 200}

def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY_INV)
    filter_kernel = np.ones((3, 3), np.uint8)
    enhanced = cv2.dilate(binary, filter_kernel, iterations=2)
    return enhanced

def detect_objects(enhanced_frame):
    contours, _ = cv2.findContours(enhanced_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        pos_x, pos_y, width, height = cv2.boundingRect(contour)
        if width > 10 and height > 10:
            objects.append((pos_x, pos_y, width, height))
    return sorted(objects, key=lambda obj: obj[0])

def calculate_timer(start_tick, elapsed):
    time_delta = time.time() - start_tick
    elapsed = time_delta * 6
    return elapsed


def calculate_safe_zone(elapsed_time):
    START_SPEED = 9
    ACCELERATION = 0.0032
    MAX_SPEED = 14

    current_speed = min(START_SPEED + ACCELERATION * elapsed_time, MAX_SPEED)
    base_safe_zone = 45 + current_speed * 10
    early_jump_buffer = -255 + current_speed * 19 + 2 * current_speed ** 2

    return int(base_safe_zone + early_jump_buffer)


def evaluate_and_act(objects, elapsed_time):
    safe_zone = calculate_safe_zone(elapsed_time)

    for obj in objects:
        x, y, w, h = obj
        center_x = x + w // 2

        if 45 < center_x < safe_zone:
            if h > 45:
                pyautogui.press("space")
                break
            elif 8 < h <= 45:
                if y < 120:
                    pyautogui.keyDown("down")
                    time.sleep(0.33)
                    pyautogui.keyUp("down")
                else:
                    pyautogui.press("space")
                break


def game_loop():
    initial_time = time.time()
    elapsed_time = 0

    with mss.mss() as capture_tool:
        while True:
            screen_frame = np.array(capture_tool.grab(SCREEN_REGION))
            processed_frame = preprocess_frame(screen_frame)
            objects = detect_objects(processed_frame)
            elapsed_time = calculate_timer(initial_time, elapsed_time)
            print(f"Текущий таймер: {elapsed_time}")
            evaluate_and_act(objects, elapsed_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    print("Игра начнется через 3 секунды...")
    time.sleep(3)
    game_loop()

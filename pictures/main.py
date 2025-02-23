import cv2

capture = cv2.VideoCapture('output.avi')
c = 0

ret, frame = capture.read()
while frame is not None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles_or_ovals = 0
    triangles = 0

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            triangles += 1
        elif len(approx) > 4:
            circles_or_ovals += 1

    if circles_or_ovals == 4 and triangles == 1:
        c += 1
        print(f"{c}) изображение найдено")
    ret, frame = capture.read()

print("\nЧисло моих изображений = ", c)


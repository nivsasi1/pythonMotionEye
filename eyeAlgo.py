import cv2
import numpy as np
import dlib
from math import hypot
import pyautogui
import time
import screeninfo

blink_threshold = 0.9  # the proportion of frames that need to be blinking to trigger the change
blink_frames = 0  # counter for the number of frames that are blinking
start_time = time.time()  # current time in seconds
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

i = 0
iris_cords = [(), (), (), (), ()]  # create a list with 6 elements, initialized to empty tuples
cords = False
overall_length = 0
right_bound = 0
overall_middle = 0
avr_shape_height = 0
blink = (100, 100, 100)
font = cv2.FONT_HERSHEY_SIMPLEX
flag1 = False

# Initialize global variables for clicks
flag = True
right_eye_blinks = 0
last_blink_time = 0
timer = None


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    # print(ratio)
    return ratio


def right_eye_closed(right_blink):
    if right_blink > blink[0]:
        #print(str(right_blink) + " is bigger " + str(blink[0]))
        return True
    return False


def left_eye_closed(left_blink):
    if left_blink > blink[1]:
        return True
    return False


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def checkWhere(cx, cy, left_blink, landmarks):
    global i, overall_length, right_bound, avr_shape_height, blink

    pixels = 20

    # Get the screen resolution
    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    # Check if the mouth is closed
    if not is_mouth_closed(landmarks):
        # Check if the overall_length have any number in(meaning the setup finished)
        if overall_length > 0:
            # Check the if left eye is closed, requirment for the mouse to go down
            if left_eye_closed(left_blink):
                # Move down left
                if cx >= overall_length * 0.65 + right_bound and pyautogui.position()[1] + pixels <= screen_height:
                    pyautogui.moveRel(-pixels, pixels)

                # Move down right
                elif cx <= overall_length * 0.35 + right_bound and pyautogui.position()[1] + pixels <= screen_height:
                    pyautogui.moveRel(pixels, pixels)

                # Move down middle
                elif overall_length * 0.35 + right_bound < cx < overall_length * 0.65 + right_bound and \
                        pyautogui.position()[1] + pixels <= screen_height:
                    pyautogui.moveRel(0, pixels)

            else:
                # Move left ups
                if cx >= overall_length * 0.65 + right_bound and cy > avr_shape_height and pyautogui.position()[
                    0] - pixels >= 0 and pyautogui.position()[1] - pixels >= 0:
                    pyautogui.moveRel(-pixels, -pixels)

                # Move left
                elif cx >= overall_length * 0.65 + right_bound and cy <= avr_shape_height and pyautogui.position()[
                    0] - pixels >= 0:
                    pyautogui.moveRel(-pixels, 0)

                # Move right ups
                elif cx <= overall_length * 0.35 + right_bound and cy > avr_shape_height and pyautogui.position()[
                    0] + pixels <= screen_width and pyautogui.position()[1] - pixels >= 0:
                    pyautogui.moveRel(pixels, -pixels)

                # Move right
                elif cx <= overall_length * 0.35 + right_bound and cy <= avr_shape_height and pyautogui.position()[
                    0] + pixels <= screen_width:
                    pyautogui.moveRel(pixels, 0)

                # Move up
                elif overall_length * 0.35 + right_bound < cx < overall_length * 0.65 + right_bound and cy > avr_shape_height and \
                        pyautogui.position()[1] - pixels >= 0:
                    pyautogui.moveRel(0, -pixels)

                # Move middle
                elif overall_length * 0.35 + right_bound < cx < overall_length * 0.65 + right_bound and cy <= avr_shape_height:
                    pass


def calculate_cords():
    global iris_cords, overall_length, avr_shape_height, right_bound, overall_middle
    left_bound = max(iris_cords[0][0][0], iris_cords[2][0][0])
    right_bound = min(iris_cords[1][0][0], iris_cords[3][0][0])
    overall_length = left_bound - right_bound

    avr_shape_height = iris_cords[0][1] + iris_cords[1][1] + iris_cords[2][1] + iris_cords[3][1]
    avr_shape_height = avr_shape_height // 4


def save_iris_coords(i, eye_cur_cords, shape):
    # print(type(shape))
    print(i, eye_cur_cords, shape)
    global iris_cords
    # i = 0 is top left
    # i = 1 is top right
    # i = 2 is left middle
    # i = 3 is right middle
    if (i < 4):
        iris_cords[i] = (eye_cur_cords, shape)
    if i == 3:
        calculate_cords()


def save_blink_size(right_eye_ratio, left_eye_ratio, blinking_ratio):
    global blink
    print(right_eye_ratio, left_eye_ratio, blinking_ratio)
    right = right_eye_ratio
    left = left_eye_ratio
    both = blinking_ratio - 0.5
    blink = (right, left, both)
    print(blink)


def detect_blink(blinking_ratio):
    if blinking_ratio > blink[2]:
        return True
    else:
        return False


def is_mouth_closed(landmarks):
    # Define the indexes of the mouth landmarks
    left_mouth_corner = 48
    right_mouth_corner = 54
    upper_lip_top = 62
    lower_lip_bottom = 66
    left_eye_left_corner = 36
    right_eye_right_corner = 45
    # Convert the landmarks to a numpy array
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Calculate the distances between landmarks
    mouth_width = np.abs(landmarks[left_mouth_corner][0] - landmarks[right_mouth_corner][0])
    mouth_height = np.abs(
        (landmarks[upper_lip_top][1] + landmarks[lower_lip_bottom][1]) / 2 - landmarks[left_mouth_corner][1])
    eye_distance = np.abs(landmarks[left_eye_left_corner][0] - landmarks[right_eye_right_corner][0])

    # Normalize the distances by the eye distance
    mouth_width /= eye_distance
    mouth_height /= eye_distance

    # Check if the mouth is closed
    if mouth_width < 0.5 and mouth_height < 0.1:
        return True
    else:
        return False


def process_blinks(right_eye_ratio, left_eye_ratio):
    global flag, right_eye_blinks, last_blink_time, timer
    now = time.time()

    if overall_length > 0:

        if timer is not None and now - timer >= 1:
            if right_eye_blinks == 1:
                left_click()
            reset_params()

        if right_eye_closed(right_eye_ratio) and not left_eye_closed(left_eye_ratio):
            if flag:
                if timer is None:
                    timer = now
                last_blink_time = now
                flag = False
                right_eye_blinks += 1
            elif now - last_blink_time > 0.2 and right_eye_blinks == 1:
                double_click()


def reset_params():
    global flag, right_eye_blinks, last_blink_time, timer
    flag = True
    right_eye_blinks = 0
    last_blink_time = 0
    timer = None

def left_click():
    pyautogui.leftClick()

def double_click():
    pyautogui.doubleClick()


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        right_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        #print(str(left_eye_ratio) + "hey" + str(right_eye_ratio))

        # print(left_eye_ratio, right_eye_ratio)
        process_blinks(right_eye_ratio, left_eye_ratio)

        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        is_blinking = detect_blink(blinking_ratio)
        if is_blinking:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))

        # Gaze detection
        right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                     (landmarks.part(37).x, landmarks.part(37).y),
                                     (landmarks.part(38).x, landmarks.part(38).y),
                                     (landmarks.part(39).x, landmarks.part(39).y),
                                     (landmarks.part(40).x, landmarks.part(40).y),
                                     (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [right_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [right_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)
        cv2.polylines(mask, [right_eye_region], True, 255, 2)

        min_x = np.min(right_eye_region[:, 0])
        max_x = np.max(right_eye_region[:, 0])
        min_y = np.min(right_eye_region[:, 1])
        max_y = np.max(right_eye_region[:, 1])
        gray_eye = left_eye[min_y: max_y, min_x: max_x]

        # 0 = black 255 = white
        # making the outside of the eye so it will be white
        _, mask = cv2.threshold(gray_eye, 0, 255, cv2.THRESH_BINARY_INV)
        # outside_of_the_eye = cv2.resize(mask, None, fx=5, fy=5)
        # cv2.imshow('outside', outside_of_the_eye)

        # the picture is now black and white
        _, threshold_eye2 = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)
        tt2 = cv2.resize(threshold_eye2, None, fx=5, fy=5)
        cv2.imshow('THRESHOLD', tt2)
        # blend both images
        threshold_eye = cv2.bitwise_not(threshold_eye2, threshold_eye2, mask=mask)
        tt = cv2.resize(threshold_eye, None, fx=5, fy=5)
        cv2.imshow('MASKANDTHRESHOLD', tt)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(threshold_eye, kernel, iterations=1)
        tt3 = cv2.resize(dilated, None, fx=5, fy=5)
        cv2.imshow('DILATED', tt3)

        inverted_img = cv2.bitwise_not(dilated)
        # print(inverted_img.shape)
        contours, hierarchy = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        moments = cv2.moments(max_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            output = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            cv2.circle(output, (cx, cy), 1, (255, 0, 0), 2)
            eye = cv2.resize(output, None, fx=10, fy=10)
            # print(cx, cy)
            eye_cur_cords = (cx, cy)
            left_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

            checkWhere(cx, inverted_img.shape[0], left_eye_ratio, landmarks)

            cv2.imshow("Output", eye)
        else:
            # cv2.putText(frame, "eye not found", (50, 150), font, 7, (255, 0, 0))
            cv2.imshow("Threshold", threshold_eye2)
            cv2.imshow("Left eye", left_eye)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    #    if data == 400:
    #        break
    if key == 27:
        break
    if key == ord('w'):
        print("w pressed")
        right_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        left_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        save_blink_size(right_eye_ratio, left_eye_ratio, blinking_ratio)

    if key == ord('s'):
        print("s pressed")
        if i < 4:
            print(i)
            try:
                save_iris_coords(i, eye_cur_cords, inverted_img.shape[0])
                i += 1
            except Exception as e:
                print(f"Error: Could not save iris coordinates for iteration {i}. Exception: {e}")

cap.release()
cv2.destroyAllWindows()

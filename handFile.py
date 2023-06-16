import cv2
import numpy as np
from sklearn.metrics import pairwise
import webbrowser
import subprocess

# global var
background = None

accumulated_weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


def calc_accum_avg(frame, accumalated_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment


def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    radius = int(0.7 * max_distance)
    circumference = (2 * np.pi * radius)
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))

        limit_points = ((circumference * 0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1
    return count


cam = cv2.VideoCapture(0)
num_frames = 0
seconds = 0
isShortCut = False
shortCutSeconds = 0
avgFingers = 0
flag = False


def checkForZeros(fingers):
    global seconds
    if fingers == 0:
        seconds += 1
    else:
        seconds = 0


def get_chrome_path():
    # Get the default Chrome executable path from the Windows registry
    try:
        output = subprocess.check_output('reg query "HKEY_CLASSES_ROOT\ChromeHTML\shell\open\command"',
                                         stderr=subprocess.DEVNULL, shell=True)
        output = output.decode('utf-8').strip()
        path_start = output.find('"') + 1
        path_end = output.rfind('"')
        chrome_path = output[path_start:path_end]
        return chrome_path
    except subprocess.CalledProcessError:
        return None


def shortCut(url, number):
    print("shortcut for " + number + " amount of fingers")
    chrome_path = get_chrome_path()
    new = 0

    if chrome_path is not None:
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
        webbrowser.get('chrome').open(url, new=new)
    else:
        print("Chrome browser not found.")


def checkForShortCut(avgFingers):
    global seconds, shortCutSeconds, flag

    seconds = 0
    shortCutSeconds = 0
    flag = False
    avg = avgFingers / 180
    if avg < 3:
        shortCut("https://docs.google.com/document/d/122rZpEg8_IN6gFCoB--73oy7zdwLe-xTd813dXvQVKs/edit?usp=sharing", avg)
    else:
        shortCut("https://www.google.co.il", avg)


while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.imshow("Finger Count", frame_copy)
    else:
        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)
            fingers = count_fingers(thresholded, hand_segment)
            if fingers >= 5:
                fingers = 5
            # check how many how long, if 0 for 3 sec
            if seconds != 180 and not flag:
                checkForZeros(fingers)
            else:
                flag = True
                seconds = 0
                shortCutSeconds += 1
                avgFingers += fingers
                cv2.putText(frame_copy, "now put the amount of fingers for shortcut", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if (shortCutSeconds == 180):
                    checkForShortCut(avgFingers)
                    avgFingers = 0

            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Thesholded", thresholded)
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)
    num_frames += 1
    cv2.imshow("Finger Count", frame_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()

'''
    File name         : person_detection.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Dimas Fadli Nugraha
    Date created      : 10/12/2018
    Date last modified: 19/12/2018
    Python Version    : 2.7
'''

from tracker import Tracker
from os import environ
import cv2
import numpy as np

MODE = environ.get('MODE', 'HAARCASCADE')
VIDEO_INPUT = environ.get('VIDEO_INPUT', 'uas_2.mp4')

if __name__ == '__main__':
    centers = []
    old_tracker = []
    current_frame = 0

    font = cv2.FONT_HERSHEY_PLAIN
    object_area_up = 12000
    object_area_down = 1000
    position_diff = 2

    # HOG Descriptor Person Detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Haarcascade Upper Body Detecter
    upperBodyCascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

    # Create object tracker
    tracker = Tracker(80, 10, 2, 1)

    # Capture Frame
    cap = cv2.VideoCapture(VIDEO_INPUT)

    while True:
        frame_limiter = 0
        centers = []
        people = []
        current_frame += 1

        # Skip 10 Frame For Better Processing
        while frame_limiter < 10:
            frame_limiter += 1
            ret, frame = cap.read()

        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))

        if MODE == 'HAARCASCADE':
            # Haarcascade Get Object
            people = upperBodyCascade.detectMultiScale(
                frame,
                scaleFactor=1.0211,
                minNeighbors=5
            )

        if MODE == 'HOG':
            # HOG Descriptor Get Object
            (people, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                     padding=(8, 8), scale=1.1)

        # Find centers of all detected objects
        found_y = 0
        for (x, y, w, h) in people:
            found_y = y
            countour_area = w * h

            if MODE == 'HAARCASCADE':
                if object_area_up > countour_area > object_area_down:
                    center = np.array([[x + w / 2], [y + h / 2]])
                    centers.append(np.round(center))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if MODE == 'HOG':
                center = np.array([[x + w / 2], [y + h / 2]])
                centers.append(np.round(center))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if centers:
            tracker.update(centers, current_frame, found_y)

            for person in tracker.tracks:
                position_lock = True
                if len(person.trace) > 1:
                    for j in range(len(person.trace) - 1):
                        # Draw trace line

                        x1 = person.trace[j][0][0]
                        y1 = person.trace[j][1][0]
                        x2 = person.trace[j + 1][0][0]
                        y2 = person.trace[j + 1][1][0]

                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

                    try:

                        trace_i = len(person.trace) - 1
                        trace_x = person.trace[trace_i][0][0]
                        trace_y = person.trace[trace_i][1][0]
                        cv2.putText(frame, 'ID: ' + str(person.track_id), (int(trace_x), int(trace_y)), font, 1,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                        for old_person in old_tracker:
                            old_trace_i = len(old_person.trace) - 2
                            old_trace_x = old_person.trace[0][0][0]
                            old_trace_y = old_person.trace[0][1][0]

                            if old_person.track_id == person.track_id and position_lock:
                                if abs(int(old_trace_x) - int(trace_x)) < position_diff and abs(
                                        int(old_trace_y) - int(trace_y)) < position_diff:
                                    print "Person With ID {} are standing still at coordinate ({},{})".format(
                                        person.track_id,
                                        trace_x, trace_y)

                                    position_lock = False

                    except:
                        pass

            old_tracker = tracker.tracks

        cv2.imshow('Person Detection {}'.format(MODE), frame)

        # Quit when escape key pressed
        if cv2.waitKey(5) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

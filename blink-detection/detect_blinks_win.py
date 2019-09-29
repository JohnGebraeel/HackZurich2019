# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

from plyer import notification

def notify():
    notification.notify(
        title='Take some rest!',
        message='You are getting TIRED. Take a rest and come back to increase your productivity!!!',
        app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
        timeout=10,  # seconds
    )

def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

def mouth_distance(mouth):
    # print(mouth)
    return dist.euclidean(mouth[3], mouth[9])

def nose_range(nose):
    # print(nose)
    return dist.euclidean(nose[0], nose[1])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
EYES_STANDARD = 20
YAWNS_STANDARD = 25
TIRED = 0

# initialize the frame counters and the total number of blinks
COUNT_EYES = 0
TOTAL = 0
SUM_BASE_RATIOS = 0
BASE_YAWN_COUNT_EYES = 200
AVERAGE_RATIO = 0.0
TIME_CYCLE = 500

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

counter = 0 # general counter 
track_eyes = [0]*TIME_CYCLE
track_yawns = [0]*TIME_CYCLE
num_eyes = 0
num_yawns = 0

# loop over frames from the video stream
while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
                break

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
                IS_YAWN = 0 
                IS_EYES = 0
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mouthStart:mouthEnd]
                nose = shape[noseStart:noseEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                mouthDIS = mouth_distance(mouth)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                noseHull = cv2.convexHull(nose)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                        COUNT_EYES += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        if COUNT_EYES >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1
                                IS_EYES = 1

                        # reset the eye frame counter
                        COUNT_EYES = 0

                if counter < BASE_YAWN_COUNT_EYES:
                    SUM_BASE_RATIOS += mouthDIS / nose_range(nose)
                elif counter == BASE_YAWN_COUNT_EYES:
                    # After BASE_YAWN_COUNT_EYES seconds, calculate average ratios 
                    AVERAGE_RATIO = SUM_BASE_RATIOS / BASE_YAWN_COUNT_EYES
                    print("average_ratio = ", AVERAGE_RATIO)
                elif counter > BASE_YAWN_COUNT_EYES and mouthDIS / nose_range(nose) > 350*AVERAGE_RATIO/100:
                    IS_YAWN = 1

                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Time: {}".format(counter), (130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if counter < BASE_YAWN_COUNT_EYES:
                    cv2.putText(frame, "Calibration time", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Mouth: {:.2f}".format(mouthDIS), (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Nose: {:.2f}".format(mouthDIS), (300, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if IS_YAWN:
                    cv2.putText(frame, "The user is yawning!!!", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # If greater than time cycle, update every second 
                if counter >= TIME_CYCLE:
                    num_eyes = num_eyes - track_eyes[0] + IS_EYES
                    num_yawns = num_yawns - track_yawns[0] + IS_YAWN
                    track_eyes.pop(0)                    
                    track_yawns.pop(0)
                    track_eyes.append(IS_EYES)
                    track_yawns.append(IS_YAWN)
                else:
                    num_eyes = num_eyes + IS_EYES
                    num_yawns = num_yawns + IS_YAWN
                    track_eyes[counter] = IS_EYES
                    track_yawns[counter] = IS_YAWN
                if num_eyes > EYES_STANDARD or num_yawns > YAWNS_STANDARD or (num_eyes + num_yawns)/2 > (EYES_STANDARD + YAWNS_STANDARD)/2:
                    cv2.putText(frame, "You're getting TIRED!!", (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    TIRED = 1
                    print("You're TIRED!")
                else:
                    TIRED = 0
                print("counter = ", counter)
                print("num eyes = ", num_eyes)
                print("num yawns = ", num_yawns)
                counter += 1
                if(counter == 100):
                    break
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
        
        if counter % 700 == 0 and TIRED:
            notify()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

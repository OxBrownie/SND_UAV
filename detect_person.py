import threading
import time
from djitellopy import Tello
import cv2 as cv
import os
from ultralytics import YOLO
from imageProcessing import Processing
import TelloControl as tc

tello = Tello()
tello.connect()

mainDir = os.getcwd()
recordingDir = os.path.join(mainDir, "Recordings")
frameDumpDir = os.path.join(recordingDir, "new")

def video_stream(tello, forward_backward_velocity, up_down_velocity, left_right_velocity, yaw_velocity):

        frame_count= 0
        tello.set_video_direction(tello.CAMERA_DOWNWARD)
        tello.streamon()  
        while True:


            drone_frame = tello.get_frame_read().frame
            crop_img = drone_frame[0:240, 0:320]
            cv.imshow("Frame", crop_img)
            if True: #add a condition statement
                frameFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_B.jpg")
                cv.imwrite(frameFile, crop_img)
                frame_count += 1

                person = YOLO("stuff.pt")
                
                results = person.predict(source=drone_frame, conf=0.5, verbose=False)
                proc = Processing(window_name="HSV view", mode=0) #need to fix
                captureFrame, centroids = proc.YOLODetect(person, results, drone_frame.copy())
                # captureFrame = captureFrame[0:240, 0:320]
                cv.imshow("stuff", captureFrame)
                telloC = (drone_frame.shape[1] // 2, drone_frame.shape[0] // 2 )
                left_right, forward, = tc.centre_person(tello, forward_backward_velocity, left_right_velocity, centroids, telloC)
                tello.send_rc_control(left_right, forward, 0, 0)
            if True:
                cv.imwrite("picture.png", drone_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break




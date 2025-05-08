############### Libraries ###############
from __future__ import print_function
import cv2
import numpy as np
from djitellopy import Tello
from imageProcessing import Processing
import time
from TelloControl import *
from ultralytics import YOLO
import os


############### Initialise ###############
# Directories
mainDir = os.getcwd()
recordingDir = os.path.join(mainDir, "Recordings")

# Window Names
captureWindow = 'Stream View'
detectionWindow = 'HSV View'
mapWindow = "Global View"

# Modes
STREAMONLY = 0
MANUALFLY = 1
COLOURCHASE = 2
SEARCHNDESTROY = 3
OBSTACLE = 4
NAVIGATE = 5

# View
WEBCAM = 0
DRONE = 1

# CV Type
CUSTOMMODE = 0
YOLOMODE = 1

# Initialise objects
model = YOLO("weights.pt")
proc = Processing(window_name=detectionWindow, mode=OBSTACLE)
tello = Tello()


############### User Define ###############
RECORD = False
NOFLY = True
view = DRONE       # WEBCAM, DRONE
mode = STREAMONLY     # STREAMONLY, MANUALFLY, COLOURCHASE, SEARCHNDESTROY, OBSTACLE
cvType = YOLOMODE   # CUSTOMMODE, YOLOMODE


############### Runtime ###############
def start(view, mode):
    """ Start Tello Drone """


    ############### Record Session ###############
    if RECORD:
        i = 1
        while True:
            session = f"{i:03d}"

            if session not in os.listdir(recordingDir):
                frameDumpDir = os.path.join(recordingDir, session)
                os.makedirs(frameDumpDir)
                print(f"Session: {session}")
                break
            else:
                i += 1

    
    ############### View ###############
    # Window (Detectwindow alreay initialised)
    cv2.namedWindow(captureWindow)

    # Select view
    if view == WEBCAM:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Overwrite mode to stream only and make sure not flying
        mode = STREAMONLY
        fly = False
        
    elif view == DRONE:
        # Drone Connection
        tello.connect(True)
        time.sleep(1)
        print(f'Battery: {tello.get_battery()}')

        # Start drone stream
        tello.streamon()
        # tello.set_video_bitrate(1)
        # tello.set_video_direction(0)

        # Fly
        if (mode == STREAMONLY) or NOFLY:
            fly = False
        else:
            fly = True
        print(f"Fly status: {fly}")
        time.sleep(5)

        if fly:
            tello.takeoff()
            # tello.move_up(60)
            # tello.flip_forward()
            tello.move_down(20)
            print("Ready")


    ############### Stream ###############
    last_time = 0
    frame_count = 0
    ready = True
    while True:
        # Frame time
        start_time = time.time()


        ############### Controls ###############
        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Start
        if cv2.waitKey(1) & 0xFF == ord('r'):
            ready = True


        ############### Actionable (10Hz) ###############
        if start_time - last_time > 0.1:
            # Get frame
            if view == WEBCAM:
                ret, frame = cap.read()

                # Check
                if not ret:
                    print("Error: Could not read frame.")
                    break

            elif view == DRONE:
                frame = tello.get_frame_read().frame
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = cv2.resize(frame, (640, 480))


            ############### Processing ###############
            # HSV filter
            detectFrame = proc.applyHSV(frame.copy())

            # Object Detect
            if mode == STREAMONLY: # Drone and webcam
                if cvType == CUSTOMMODE:
                    captureFrame, centroids = proc.poleDetect(detectFrame, frame.copy())

                elif cvType == YOLOMODE:
                    results = model([frame])
                    # results = model.predict(source=frame, conf=0.5, verbose=False)
                    print(results)
                    captureFrame, centroids = proc.YOLODetect(model, results, frame.copy())
                    
            elif mode == OBSTACLE:
                if cvType == CUSTOMMODE:
                    detectFrame = proc.getPoleMask(detectFrame)
                    captureFrame, centroids = proc.poleDetect(detectFrame, frame.copy())

                elif cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.5, verbose=False)
                    captureFrame, centroids = proc.YOLODetect(model, results, frame.copy())
                    
            elif mode == COLOURCHASE:
                captureFrame, centroids = proc.poleDetect(detectFrame, frame.copy())
                

            # Draw buttons
            proc.draw_buttons(detectFrame)


            ############### Show Frames ###############
            cv2.imshow(captureWindow, captureFrame)
            cv2.imshow(detectionWindow, detectFrame)


            ############### Record ###############
            if RECORD:
                detectFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_A.jpg")
                frameFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_B.jpg")
                # cv2.imwrite(detectFile, captureFrame)
                cv2.imwrite(frameFile, frame)
                frame_count += 1


            ############### Mode Execution ###############
            telloC = (frame.shape[1] // 2, frame.shape[0] // 2 ) # x,y
            if mode == STREAMONLY:
                proc.printHSV()
                left_right = forward = up_down = yaw = 0
            
            elif mode == OBSTACLE:
                proc.printHSV()
                if centroids is not []:
                    left_right, forward, up_down, yaw = navigate_through_poles(centroids, telloC)
                    # yaw = 0
                else:
                    left_right = forward = up_down = yaw = 0

            elif mode == COLOURCHASE:
                if centroids is not []:
                    left_right, forward, up_down, yaw = chase(centroids, telloC)
                    yaw = 0
                else:
                    left_right = forward = up_down = yaw = 0

            # Send yaw control to Tello
            if fly:
                # Wait for ready command
                if not ready:
                    continue
                else:
                    # Battery (to keep awake)
                    print(f'Battery: {tello.get_battery()}')

                tello.send_rc_control(left_right, forward, up_down, yaw)
            else:
                print(f"Controls: {[-left_right, forward, up_down, yaw]}")
    

            ############### Stream Loop ###############
            last_time = time.time()


    ############### Exit ###############
    if view == WEBCAM:
        cap.release() 

    elif view == DRONE:
        tello.streamoff()
        if fly:
            tello.land()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Start')

    # startWebcam()
    start(view, mode)
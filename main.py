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
from map import *
from threading import Thread, Event


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
OBSTACLE = 3
SEEK = 4
SEARCH = 5
CAPTURE = 6
HOME = 7
LAND = 8

# View
WEBCAM = 0
DRONE = 1

# CV Type
CUSTOMMODE = 0
YOLOMODE = 1

# World Coordinates
searchCoordinates = (0, 600)
targetCoordinates = (None, None)
landCoordinates = (250, 0)

# Threading
global stop_event, map_thread

# Initialise objects
model = YOLO("SearchUAV_V02.pt")
proc = Processing(window_name=detectionWindow, mode=OBSTACLE)
tello = Tello()
map = Map2D(search=searchCoordinates, land=landCoordinates)


############### User Define ###############
RECORD = False
NOFLY = False
view = DRONE          # WEBCAM, DRONE
mode = OBSTACLE       # Manually fix mode
cvType = YOLOMODE     # CUSTOMMODE, YOLOMODE
thread = True


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
    # Object detect
    if cvType == YOLOMODE:
        cv2.destroyWindow(detectionWindow)
        detectWindow = False
    else:
        detectWindow = True
    
    # Stream 
    cv2.namedWindow(captureWindow)

    # Map
    if thread:
        cv2.namedWindow(mapWindow)


    ############### Camera ###############
    if view == WEBCAM:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Overwrite mode to stream only and make sure not flying
        mode = STREAMONLY
        fly = False
        
    elif view == DRONE:
        # Drone Connection
        tello.connect(True)
        time.sleep(1)
        print(f'Battery: {tello.get_battery()}, Temperature: {tello.get_temperature()}')

        # Start drone stream
        tello.streamon()

        # Fly
        if (mode == STREAMONLY) or NOFLY:
            fly = False
        else:
            fly = True
        print(f"Fly status: {fly}")
        time.sleep(1)

        if fly:
            tello.send_rc_control(0, 0, 0, 0)
            tello.takeoff()
            time.sleep(1)
            # tello.send_rc_control(0, 0, -20, 0)
            # while True:
            #     height = tello.get_height()
            #     if height < 30: break
            print("Ready")

    
    ############### Threading ###############
    if thread and (view == DRONE):
        stop_event = Event()
        map_thread = Thread(target=update_map_loop, args=(tello, map, stop_event))
        map_thread.start()


    ############### Stream ###############
    # Initialise
    last_time = 0
    frame_count = 0
    centroids = []

    # Loop
    while True:
        # Frame time
        start_time = time.time()
        dt = start_time - last_time


        ############### Controls ###############
        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        ############### Actionable (10Hz) ###############
        if dt > 0.1:
            ############### Frame ###############
            # Webcam
            if view == WEBCAM:
                ret, frame = cap.read()

            # Drone
            elif view == DRONE:
                frame = tello.get_frame_read().frame
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize
            frame = cv2.resize(frame, (640, 480))


            ############### Processing ###############
            # HSV filter
            if detectWindow: detectFrame = proc.applyHSV(frame.copy())

            # Object Detect
            if mode == STREAMONLY:
                if cvType == CUSTOMMODE:
                    captureFrame, centroids = proc.objectDetect(detectFrame, frame.copy())

                elif cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, centroids = proc.YOLODetectPoles(model, results, frame.copy())
            
            elif mode == COLOURCHASE:
                captureFrame, centroids = proc.objectDetect(detectFrame, frame.copy())
                    
            elif mode == OBSTACLE:
                if cvType == CUSTOMMODE:
                    detectFrame = proc.getPoleMask(detectFrame)
                    captureFrame, centroids = proc.poleDetect(detectFrame, frame.copy())

                elif cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, centroids = proc.YOLODetectPoles(model, results, frame.copy())
                    
            # Draw buttons
            if detectWindow: proc.draw_buttons(detectFrame)


            ############### Show Frames ###############
            # Object detect
            if detectWindow:
                cv2.imshow(detectionWindow, detectFrame)

            # View
            cv2.imshow(captureWindow, captureFrame)

            # Map
            if fly:
                if thread:
                    yaw = tello.get_yaw()
                    yaw = np.radians(yaw)
                    map_frame = map.draw(yaw=yaw)
                    cv2.imshow(mapWindow, map_frame)
                

            ############### Record ###############
            if RECORD:
                detectFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_A.jpg")
                frameFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_B.jpg")
                cv2.imwrite(detectFile, captureFrame)
                cv2.imwrite(frameFile, frame)
                frame_count += 1


            ############### Mode Execution ###############
            # Initialise
            telloCentre = (frame.shape[1] // 2, frame.shape[0] // 2 )
            left_right = forward_back = up_down = yawleft_right = 0

            # Modes
            if mode == STREAMONLY:
                # Prints
                if detectWindow: proc.printHSV()

            elif mode == COLOURCHASE:
                # Prints
                if detectWindow: proc.printHSV()

                # Controls
                left_right, forward_back, up_down, yawleft_right = chase(centroids, telloCentre)
            
            elif mode == OBSTACLE:
                # Prints
                if detectWindow: proc.printHSV()

                # Controls
                left_right, forward_back = navigate_through_poles(centroids, telloCentre, dt)
                up_down = yawleft_right = 0

            
            # Commands
            if fly:
                # Get general state
                print(f'Battery: {tello.get_battery()}, Temperature: {tello.get_temperature()}')

                # Send control
                tello.send_rc_control(left_right, forward_back, up_down, yawleft_right)

            else:
                print(f"Controls: {[left_right, forward_back, up_down, yawleft_right]}")
    

            ############### End Actionable Loop ###############
            last_time = time.time()


    ############### Exit ###############
    # Camera
    if view == WEBCAM:
        cap.release() 

    elif view == DRONE:
        tello.streamoff()
        if fly:
            tello.land()

    # Windows
    cv2.destroyAllWindows()

    # Thread
    if thread:
        stop_event.set()
        map_thread.join()


############### Main Script ###############
if __name__ == "__main__":
    print('Start')

    start(view, mode)
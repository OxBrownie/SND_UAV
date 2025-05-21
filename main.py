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
import detect_person as dp

############### Initialise ###############
# Directories
mainDir = os.getcwd()
recordingDir = os.path.join(mainDir, "Recordings")
captureDir = os.path.join(mainDir, "Capture")

# Window Names
captureWindow = 'Stream View'
detectionWindow = 'HSV View'
mapWindow = "Global View"

# Modes
STREAMONLY = 0
MANUALFLY = 1
COLOURCHASE = 2
STARTUP = 3
OBSTACLE = 4
SEEK = 5
SEARCH = 6
CAPTURE = 7
HOME = 8
LAND = 9

# View
WEBCAM = 0
DRONE = 1

# CV Type
CUSTOMMODE = 0
YOLOMODE = 1

# World Coordinates
searchCoordinates = (0, 600)
searchCoordinates = (0, 600)
targetCoordinates = (None, None)
landCoordinates = (250, 0)

# Threading
global stop_event, map_thread

# Initialise objects
model = YOLO("SearchUAV_V04.pt")
proc = Processing(window_name=detectionWindow, mode=OBSTACLE)
tello = Tello()
map = Map2D(search=searchCoordinates, land=landCoordinates)


############### User Define ###############
RECORD = True
NOFLY = False
view = DRONE          # WEBCAM, DRONE
ORIENTATION = 0
mode = STARTUP       # Manually fix mode
cvType = YOLOMODE     # CUSTOMMODE, YOLOMODE
thread = True
transit = True
switch = False


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
        time.sleep(2)
        print(f'Battery: {tello.get_battery()}, Temperature: {tello.get_temperature()}')

        # Camera direction
        if mode in [SEARCH, CAPTURE, LAND]:
            ORIENTATION = 1   
        else:
            ORIENTATION = 0
        tello.set_video_direction(ORIENTATION)
        time.sleep(1)
            
        # Start drone stream
        tello.streamon()
        time.sleep(1)


        ############### Take Off ###############
        if (mode == STREAMONLY) or NOFLY:
            fly = False
        else:
            fly = True
        print(f"Fly status: {fly}")
        time.sleep(1)

        if fly:
            tello.send_rc_control(0, 0, 0, 0)
            tello.takeoff()
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
    target = []
    LZ = []
    waypoint_index = 0
    found = False
    inView = True
    thresh_capture = 0

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
                frame = cap.read()

            # Drone
            elif view == DRONE:
                frame = tello.get_frame_read().frame
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize
            if ORIENTATION == 0:
                frame = cv2.resize(frame, (640, 480))
            else:
                frame = frame[0:240, 0:320]


            ############### Processing ###############
            # HSV filter
            if detectWindow: detectFrame = proc.applyHSV(frame.copy())

            # Non active Modes
            if mode == STREAMONLY:
                if cvType == CUSTOMMODE:
                    captureFrame, centroids = proc.objectDetect(detectFrame, frame.copy())

                elif cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, targets = proc.YOLODetectTarget(model, results, frame.copy())
            
            elif mode == COLOURCHASE:
                captureFrame, centroids = proc.objectDetect(detectFrame, frame.copy())

            # Active modes 
            if mode == STARTUP:
                captureFrame = frame.copy()
                
            elif mode == OBSTACLE:
                if cvType == CUSTOMMODE:
                    detectFrame = proc.getPoleMask(detectFrame)
                    captureFrame, centroids = proc.poleDetect(detectFrame, frame.copy())
                
                elif cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, centroids = proc.YOLODetectPoles(model, results, frame.copy())
            
            elif mode == SEEK:
                captureFrame = frame.copy()

            elif mode == SEARCH:
                if cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, target = proc.YOLODetectTarget(model, results, frame.copy())

            elif mode == CAPTURE:
                if cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, target = proc.YOLODetectTarget(model, results, frame.copy())

            elif mode == CAPTURE:
                if cvType == YOLOMODE:
                    person = YOLO("stuff.pt")
                    results = person.predict(source=frame, conf=0.5, verbose=False)
                    proc = Processing(window_name="HSV view", mode=0) #need to fix
                    captureFrame, centroids = proc.YOLODetectPoles(person, results,frame.copy())

            elif mode == HOME:
                captureFrame = frame.copy()

            elif mode == LAND:
                if cvType == YOLOMODE:
                    results = model.predict(source=frame, conf=0.1, verbose=False)
                    captureFrame, LZ = proc.YOLODetectTarget(model, results, frame.copy())


            ############### Show Frames ###############
            # Draw buttons
            if detectWindow: proc.draw_buttons(detectFrame)

            # Object detect
            if detectWindow:
                cv2.imshow(detectionWindow, detectFrame)

            # View
            cv2.imshow(captureWindow, captureFrame)

            # Map
            if fly:
                if thread:
                    yaw = 0 # Never rotate the drone
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

            # Non-active modes
            if mode == STREAMONLY:
                # Prints
                if detectWindow: proc.printHSV()

            elif mode == COLOURCHASE:
                # Prints
                if detectWindow: proc.printHSV()

                # Controls
                left_right, forward_back, up_down, yawleft_right = chase(centroids, telloCentre)
            
            # Active Modes
            if mode == STARTUP:
                up_down = -20
                height = tello.get_height()

                # State change
                if height < 35:
                    mode = OBSTACLE

            elif mode == OBSTACLE:
                # Prints
                if detectWindow: proc.printHSV()

                # Controls
                left_right, forward_back, inView = navigate_through_poles(centroids, telloCentre, dt)
                up_down = yawleft_right = 0

                # State change
                if not inView:
                    mode = SEEK

            elif mode == SEEK:
                # Current position
                drone_pos = map.getDrone()

                # Controls
                left_right, forward_back, reached = navigate_to(drone_pos, searchCoordinates, yaw, threshold=10, speed_limit=20)

                # State change
                if reached:
                    mode = SEARCH

                    # Switch Camera
                    if switch:
                        ORIENTATION = 1
                        tello.set_video_direction(ORIENTATION)

            elif mode == SEARCH:
                # Print
                print(f"Waypoint: {waypoint_index}")

                # Set Waypoints
                waypoints = [(0, 100),   # 1m up
                          (-100, 0),     # 1m left
                          (0, -200),     # 2m down
                          (200, 0),      # 2m right
                          (0, 200),      # 2m up
                          (-100, 0)]     # 1m left

                # Coordinates
                drone_pos = map.getDrone()
                waypoint = waypoints[waypoint_index]

                # Controls
                left_right, forward_back, reached = navigate_to(drone_pos, waypoint, yaw, threshold=10, speed_limit=20)

                # Waypoint update
                if reached:
                    waypoint_index += 1
                    if waypoint_index >= len(waypoints):
                        left_right = forward_back = 0
                
                # State change
                if target is not []:
                    mode = CAPTURE

            elif mode == CAPTURE:
                if len(target) != 0:
                    left_right, forward_back, aligned = align_target(target, telloCentre, dt)
                else:
                    aligned = False

                # State change
                if aligned and transit:
                    # Get coordinates
                    drone_pos = map.getDrone()
                    print(drone_pos)

                    # Take picture
                    frameCapture = os.path.join(captureDir, f"FOUND_{drone_pos}.jpg")
                    cv2.imwrite(frameCapture, captureFrame)

                    # Change
                    mode = HOME

            elif mode == HOME:
                # Current position
                drone_pos = map.getDrone()

                # Controls
                left_right, forward_back, reached = navigate_to(drone_pos, landCoordinates, yaw, threshold=10, speed_limit=30)

                # State change
                if reached:
                    mode = LAND

            elif mode == LAND:
                up_down = -20
                if len(target) != 0:
                    left_right, forward_back, aligned = align_target(target, telloCentre, dt)

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
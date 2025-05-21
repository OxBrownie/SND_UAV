############### Libraries ###############
from __future__ import print_function
import cv2
import numpy as np
from djitellopy import Tello
from imageProcessing import Processing
import time
from TelloControl import *
from ultralytics import YOLO
from map import *
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
SEEK = 3
OBSTACLE = 4
CAPTURE = 5
HOME = 6
LAND = 7

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
correction_x = 17
correction_y = 0.8

# Initialise objects
model = YOLO("SearchUAV_V02.pt")
proc = Processing(window_name=detectionWindow, mode=OBSTACLE)
tello = Tello()
map = Map2D(search=searchCoordinates, land=landCoordinates)


############### User Define ###############
RECORD = False
NOFLY = False
view = DRONE       # WEBCAM, DRONE
mode = SEEK     # 
cvType = CUSTOMMODE   # CUSTOMMODE, YOLOMODE


############### Runtime ###############
def start(view, mode):
    """ Start Tello Drone """


    ############### View  ###############
    # Window
    cv2.destroyWindow(detectionWindow)
    cv2.namedWindow(captureWindow)
    cv2.namedWindow(mapWindow)

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
        # tello.move_down(20)
        print("Ready")


    ############### Stream ###############
    last_time = 0
    ready = True
    sys_time = time.time()
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
            # Get frame
            frame = tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))


            ############### Processing ###############
            


            ############### Map Position ###############
            # Tello state
            yaw = tello.get_yaw()
            yaw = np.radians(yaw)
            vx = tello.get_speed_x()
            vy = tello.get_speed_y()

            # Position change        
            dx = vy * dt * correction_y
            dy = vx * dt * correction_x

            # Update
            map.update_drone_position(dx=dx, dy=dy)
            mapFrame = map.draw(yaw=yaw)


            ############### Show Frames ###############
            cv2.imshow(captureWindow, frame)
            cv2.imshow(mapWindow, mapFrame)


            ############### Mode Execution ###############
            # Initialise
            left_right = forward_back = up_down = yawleft_right = 0
            drone_pos = map.getDrone()

            # Modes
            if mode == SEEK:
                left_right, forward_back, reached = navigate_to(drone_pos, searchCoordinates, yaw, threshold=10, speed_limit=20)
                if reached: mode = HOME 
            elif mode == HOME:
                left_right, forward_back, reached = navigate_to(drone_pos, landCoordinates, yaw, threshold=10, speed_limit=30)
                if reached: mode = LAND
            
            # Control
            tello.send_rc_control(left_right, forward_back, up_down, yawleft_right)


            ############### Stream Loop End ###############
            last_time = time.time()

    
    ############### Exit ###############
    tello.streamoff()
    if fly:
        tello.land()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Start')

    # startWebcam()
    start(view, mode)

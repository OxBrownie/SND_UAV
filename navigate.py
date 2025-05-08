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
model = YOLO("yolov8n.pt")
proc = Processing(window_name=detectionWindow, mode=OBSTACLE)
tello = Tello()
map = Map2D()


############### User Define ###############
RECORD = False
NOFLY = False
view = DRONE       # WEBCAM, DRONE
mode = OBSTACLE     # STREAMONLY, MANUALFLY, COLOURCHASE, SEARCHNDESTROY, OBSTACLE
cvType = CUSTOMMODE   # CUSTOMMODE, YOLOMODE


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


    ############### View  ###############
    # Window
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
        dt = start_time - last_time
        if dt > 0.1:
            # Get frame
            frame = tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))


            ############### Processing ###############
            # fake example:
            poles = [(50, 100), (-30, 80)]  # relative to drone: (x=left/right, y=forward)


            ############### Map Position ###############
            # Get from Tello
            yaw = tello.get_yaw()
            yaw = np.radians(yaw)
            vx = tello.get_speed_x() / 100.0  # cm/s → m/s
            vy = tello.get_speed_y() / 100.0  

            # Position change
            vx_global =  vx * np.cos(yaw) - vy * np.sin(yaw)
            vy_global =  vx * np.sin(yaw) + vy * np.cos(yaw)          
            dx = vx_global * dt
            dy = vy_global * dt

            # Update
            map.update_drone_position(dx=dx, dy=dy)
            mapFrame = map.draw(poles=poles, yaw=np.yaw)


            ############### Show Frames ###############
            cv2.imshow(captureWindow, frame)
            cv2.imshow("Map View", mapFrame)


            ############### Record ###############
            if RECORD:
                mapFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_MAP.jpg")
                detectFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_A.jpg")
                frameFile = os.path.join(frameDumpDir, f"frame_{frame_count:06d}_B.jpg")
                # cv2.imwrite(detectFile, captureFrame)
                cv2.imwrite(frameFile, frame)
                cv2.imwrite(mapFile, mapFrame)
                frame_count += 1


            ############### Mode Execution ###############
            # Do Nothing here for now


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
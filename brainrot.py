import threading
import time
from djitellopy import Tello

tello = Tello()
tello.connect()
tello.send_rc_control(0, 0, 0, 0)  # Stop all movement
tello.streamoff()

# Lock to protect Tello commands
tello_lock = threading.Lock()

def fly(tello):
    with tello_lock:
        tello.takeoff()
    time.sleep(2)

    with tello_lock:
        # tello.move_forward(500)
        time.sleep(1)
        tello.send_rc_control(0, 50, 0, 0)
    time.sleep(2)

    with tello_lock:
        tello.send_rc_control(0, 0, 0, 0)  # Stop all movement
        tello.land()

def log_velocity(tello, stop_event):
    while not stop_event.is_set():
        with tello_lock:
        #     vx = tello.get_speed_x()
        #     vy = tello.get_speed_y()
        #     vz = tello.get_speed_z()
            vx = tello.get_speed_x() # cm/s → m/s
            vy = tello.get_speed_y() 
            vz = 0
            battery = tello.get_battery()
        print(f'Battery: {battery}')
        print(f"Velocity: vx={vx} cm/s, vy={vy} cm/s, vz={vz} cm/s")
        time.sleep(0.5)

# Event to signal stop
stop_event = threading.Event()

# Threads
t1 = threading.Thread(target=fly, args=(tello,))
t2 = threading.Thread(target=log_velocity, args=(tello, stop_event))

# Start both
t2.start()
t1.start()

# Wait for flying to finish
t1.join()

# Stop logging
stop_event.set()
t2.join()

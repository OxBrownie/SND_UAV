############### Libraries ###############
import cv2
import numpy as np
import time


############### Threaded Update Position ###############
def update_map_loop(tello, map, stop_event, rc_state, rc_lock):
    """
    Runs in a separate thread. Periodically updates drone position on the map.
    Uses manually scaled velocity from Tello state.
    """
    update_interval = 0.1  # 10 Hz
    last_time = time.time()

    ############### Map Loop ###############
    while not stop_event.is_set():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        try:
            ############### Tello Speeds ###############
            with rc_lock:
                vx = rc_state['left_right']
                vy = rc_state['forward_back']

            # # Forward corrections
            # if vx > 0:
            #     vx *= -12.5
            # else:
            #     vx *= -17
            
            # # Lateral corrections
            # if vy > 0:
            #     vy *= 17
            # else:
            #     vy *= 17

            ############### Position Change ###############
            # Vertical change
            dy = vy*dt

            # Lateral change
            dx = vx*dt

            # Check
            # print(f"[Velocity] vx: {vx}, vy: {vy} cm/s → dx: {dx:.1f}, dy: {dy:.1f} cm")

            # Update position on map
            map.update_drone_position(dx=dx, dy=dy)

        except Exception as e:
            print("Manual update failed:", e)

        time.sleep(update_interval)


############### World Map ###############
class Map2D:
    def __init__(self, size=(1500, 1500), search=None, land=None):
        self.size = size
        self.map = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        self.scale = 1
        self.center = (size[0] // 2, size[1] // 2)
        self.drone_pos = self.center
        self.drone_pos_cm = (0.0, 0.0)
        self.path = [self.drone_pos]
        self.direction = 0

        # Takeoff position
        cv2.rectangle(self.map,
                    (self.center[0] - 5, self.center[1] - 5),
                    (self.center[0] + 5, self.center[1] + 5),
                    (0, 0, 0), -1)
        cv2.putText(self.map, "Takeoff", (self.center[0] + 10, self.center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Search position
        if search is not None:
            self.searchCoords = search
            sx, sy = self.cm_to_px(*search)
            cv2.circle(self.map, (sx, sy), 6, (0, 0, 255), -1)
            cv2.putText(self.map, "Search", (sx + 10, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Land position
        if land is not None:
            self.landCoords = land
            lx, ly = self.cm_to_px(*land)
            cv2.rectangle(self.map,
                        (lx - 5, ly - 5),
                        (lx + 5, ly + 5),
                        (0, 0, 0), -1)
            cv2.putText(self.map, "Land", (lx + 10, ly + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
    def cm_to_px(self, x_cm, y_cm):
            x_px = int(self.center[0] + x_cm * self.scale)
            y_px = int(self.center[1] - y_cm * self.scale)
            return (x_px, y_px)


    def update_drone_position(self, dx, dy):
        x_cm, y_cm = self.drone_pos_cm
        self.drone_pos_cm = (x_cm + dx, y_cm + dy)
        px_pos = self.cm_to_px(*self.drone_pos_cm)
        self.path.append(px_pos)

    def set_position(self, pos_cm):
        self.drone_pos_cm = pos_cm
        px_pos = self.cm_to_px(*self.drone_pos_cm)
        self.path.append(px_pos)

    def get_position(self):
        return self.drone_pos_cm

    def getDrone(self):
        return self.drone_pos_cm

    def draw(self, yaw=0):
        display = self.map.copy()

        # Drone path
        for i in range(1, len(self.path)):
            pt1 = tuple(map(int, self.path[i - 1]))
            pt2 = tuple(map(int, self.path[i]))
            cv2.line(display, pt1, pt2, (255, 0, 0), 2)

        x_px, y_px = self.cm_to_px(*self.drone_pos_cm)

        # Triangle drone shape
        self.direction += yaw
        size = 20
        tip_x = x_px + size * np.sin(yaw)
        tip_y = y_px - size * np.cos(yaw)

        left_x = x_px + size * 0.4 * np.sin(yaw + np.pi / 2)
        left_y = y_px - size * 0.4 * np.cos(yaw + np.pi / 2)

        right_x = x_px + size * 0.4 * np.sin(yaw - np.pi / 2)
        right_y = y_px - size * 0.4 * np.cos(yaw - np.pi / 2)

        triangle = np.array([
            [int(tip_x), int(tip_y)],
            [int(left_x), int(left_y)],
            [int(right_x), int(right_y)]
        ])
        cv2.drawContours(display, [triangle], 0, (0, 0, 255), -1)

        cv2.putText(display, f"{self.drone_pos_cm[0]:.2f}, {self.drone_pos_cm[1]:.2f}", (x_px, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return display
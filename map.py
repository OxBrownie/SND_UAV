import cv2
import numpy as np


class Map2D:
    def __init__(self, size=(600, 600)):
        self.size = size
        self.map = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        self.scale = 5  # pixels per cm (or meters depending on your use)

        # Start drone at center
        self.drone_pos = (size[0] // 2, size[1] // 2)
        self.path = [self.drone_pos]

    def update_drone_position(self, dx, dy):
        x, y = self.drone_pos
        new_x = int(x + dx * self.scale)
        new_y = int(y - dy * self.scale)  # Y-axis inverted to match visual intuition
        self.drone_pos = (new_x, new_y)
        self.path.append(self.drone_pos)

    def draw(self, poles=None, yaw=0):
        display = self.map.copy()

        # Draw path
        for i in range(1, len(self.path)):
            cv2.line(display, self.path[i - 1], self.path[i], (200, 200, 200), 2)

        # Draw poles
        if poles:
            for pole in poles:
                px = int(self.drone_pos[0] + pole[0] * self.scale)
                py = int(self.drone_pos[1] - pole[1] * self.scale)
                cv2.circle(display, (px, py), 5, (0, 0, 255), -1)

        # Draw drone as a triangle with orientation
        x, y = self.drone_pos
        length = 10
        angle = -yaw + np.pi  # convert yaw to map orientation
        tip = (int(x + length * np.cos(angle)), int(y - length * np.sin(angle)))
        left = (int(x + length * np.cos(angle + 2.5)), int(y - length * np.sin(angle + 2.5)))
        right = (int(x + length * np.cos(angle - 2.5)), int(y - length * np.sin(angle - 2.5)))
        cv2.drawContours(display, [np.array([tip, left, right])], 0, (0, 255, 0), -1)

        return display


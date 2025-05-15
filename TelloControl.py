############### Libraries ###############
import time
import cv2
import numpy as np


############### PID Control ###############
prev_error = 0
integral = 0

def chase(centroids, telloC):
    """Chase the closest object (centroids[0]) using yaw, left/right, and up/down control."""
    print(centroids)
    if len(centroids) >= 1:
        target = centroids[0]
    else:
        return 0,0,0,0
    
    # Default RC commands
    forward_backward_velocity = 0  # Not moving forward
    up_down_velocity = 0
    left_right_velocity = 0
    yaw_velocity = 0

    if centroids:
        target_x = target[0]
        target_y = target[1]
        current_x = telloC[0]
        current_y = telloC[1]

        # Calculate errors
        error_x = target_x - current_x
        error_y = target_y - current_y

        # Tuning parameters
        Kp_yaw = 0.1
        Kp_lr = 0.1
        Kp_ud = 0.1

        # Proportional control
        yaw_velocity = int(Kp_yaw * error_x)
        left_right_velocity = int(Kp_lr * error_x)
        up_down_velocity = int(Kp_ud * (-error_y))  # Negative to go up when object is higher

        # Clamp values to Tello's range (-100 to 100)
        yaw_velocity = max(-100, min(100, yaw_velocity))
        left_right_velocity = max(-100, min(100, left_right_velocity))
        up_down_velocity = max(-100, min(100, up_down_velocity))

    return left_right_velocity, forward_backward_velocity, up_down_velocity, -yaw_velocity
  
def navigate_through_poles(centroids, telloCentre, dt):
    """Navigate using PID control for left-right movement based on detected poles."""
    global prev_error, integral
    LOCKOUT = False

    # Default velocity
    left_right = 0
    forward_back = 20

    # PID constants
    Kp = 0.0001   # Reduce P gain to prevent overshooting
    Ki = 0.005  # Small I gain to prevent drifting
    Kd = 0.05   # Reduce D gain to smooth movements

    # Frame centre
    mid_x, mid_y = telloCentre  

    # Navigate between two cloests poles
    if len(centroids) >= 2:
        c1, c2 = centroids[0], centroids[1]
        target_x = (c1[0] + c2[0]) / 2
        error = target_x - mid_x

        # PID control
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        output = Kp * error + Ki * integral + Kd * derivative
        prev_error = error

        left_right = int(output)

        if abs(error) < 20:
            print("LOCKOUT")
            LOCKOUT = False

    elif len(centroids) == 1:
        pole_x = centroids[0][0]
        error = mid_x - pole_x
        abs_error = abs(error)

        if abs_error < 50:
            # Pole directly ahead, strong immediate reaction
            left_right = 10 if error > 0 else -10
        else:
            # Inversely proportional strafe (closer pole = stronger avoidance)
            k = 10000  # Gain factor, tweak as needed
            output = k / abs_error

            # Cap output between -10 and 10
            left_right = int(np.clip(np.sign(error) * output, -10, 10))

    else:
        left_right = forward_back = 0

    return left_right, forward_back, LOCKOUT

def navigate_to(current_pos, target_pos, yaw, threshold=10, speed_limit=30):
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = np.hypot(dx, dy)

    if distance < threshold:
        return 0, 0

    # Direction vector
    direction_x = dx / distance
    direction_y = dy / distance

    # Convert to drone-relative frame
    rel_x = direction_x * np.cos(-yaw) - direction_y * np.sin(-yaw)
    rel_y = direction_x * np.sin(-yaw) + direction_y * np.cos(-yaw)

    forward_back = int(rel_y * min(speed_limit, distance))
    left_right = int(rel_x * min(speed_limit, distance))

    return left_right, forward_back
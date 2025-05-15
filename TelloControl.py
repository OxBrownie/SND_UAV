############### Libraries ###############
import time
import cv2
import numpy as np


############### Global Variables ###############
# PID
prev_error = 0
integral = 0

# Gap locking
locked_gap = None           # Stores (c1, c2) once locked
gap_lock_timer = 0          # Counts stable frames
GAP_LOCK_DURATION = 5       # Lock duration in frames

# Temporal smoothing
smoothed_output = 0         # For temporal smoothing
SMOOTH_ALPHA = 0.6          # Weight for exponential smoothing


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
    ############### Initialise ###############
    # PID
    global prev_error, integral
    global locked_gap, gap_lock_timer, smoothed_output

    # PID constants
    Kp = 0.0001   # Reduce P gain to prevent overshooting
    Ki = 0.005  # Small I gain to prevent drifting
    Kd = 0.05   # Reduce D gain to smooth movements

    # Default velocity
    left_right = 0
    forward_back = 15

    # Frame centre
    mid_x, mid_y = telloCentre  


    ############### Two Pole Thread ###############
    if len(centroids) >= 2:
        # Locked gap
        if gap_lock_timer > 0 and locked_gap is not None:
            c1, c2 = locked_gap
            gap_lock_timer -= 1

        # Lock to new pair
        else:
            c1, c2 = centroids[0], centroids[1]
            locked_gap = (c1, c2)
            gap_lock_timer = GAP_LOCK_DURATION

        # Get target
        target_x = (c1[0] + c2[0]) / 2
        error = target_x - mid_x

        # PID Control
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        raw_output = Kp * error + Ki * integral + Kd * derivative
        prev_error = error

        # Temporal smoothing response
        smoothed_output = SMOOTH_ALPHA * smoothed_output + (1 - SMOOTH_ALPHA) * raw_output
        left_right = int(smoothed_output)


    ############### Single Pole Avoidance ###############
    elif len(centroids) == 1:
        # New speed
        forward_back = 10

        # Initialise 
        pole_x = centroids[0][0]
        error = mid_x - pole_x
        sign = np.sign(error)
        abs_error = max(abs(error), 10)

        # Inverse proportional Gain
        k = 1
        output = k / abs_error

        # Response
        left_right = int(np.clip(sign * output, -20, 20))

        # Reset lock
        gap_lock_timer = 0
        locked_gap = None
        smoothed_output = 0


    ############### No Poles ###############
    else:
        # Response
        forward_back = left_right = 0

        # Reset lock
        gap_lock_timer = 0
        locked_gap = None
        smoothed_output = 0

    return left_right, forward_back

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
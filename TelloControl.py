############### Libraries ###############
import time
import cv2
import numpy as np


############### Global Variables ###############
# PID
prev_error = 0
integral = 0

prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0

# Long term error
error_buffer = []  # [(error_x, error_y), ...]
ALIGNMENT_WINDOW = 30  # Number of frames to consider (~2 sec if running at 10Hz)


# Gap locking
locked_centroids = None
lock_frames_remaining = 0


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
    global prev_error, integral
    global locked_centroids, lock_frames_remaining

    # PID constants
    Kp = 0.005
    Ki = 0.005
    Kd = 0.07

    # Default velocities
    left_right = 0
    forward_back = 15

    # Frame centre
    mid_x, mid_y = telloCentre

    ############### Two Poles ###############
    if len(centroids) >= 2:
        # Update lock if expired
        if lock_frames_remaining == 0:
            locked_centroids = (centroids[0], centroids[1])
            lock_frames_remaining = 5
        else:
            lock_frames_remaining -= 1

        c1, c2 = locked_centroids
        target_x = (c1[0] + c2[0]) / 2
        error = target_x - mid_x
        sign = np.sign(error)

        # PID control
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        prev_error = error

        # if abs(error) < 5:
        #     output = 0
        # else:
        output = Kp * error + Ki * integral + Kd * derivative

        # Apply PID output
        # left_right = int(output)
        left_right = int(np.clip(sign * output, -40, 40))

        # Flag
        inView = True


    ############### Single Pole Avoidance ###############
    elif len(centroids) == 1:
        forward_back = 10

        pole_x = centroids[0][0]
        error = mid_x - pole_x
        sign = np.sign(error)
        abs_error = max(abs(error), 10)

        # Inverse proportional gain
        k = 1500
        output = k / abs_error
        left_right = int(np.clip(sign * output, -25, 25))

        # Reset PID
        prev_error = 0
        integral = 0

        # Flag
        inView = True

    ############### No Poles ###############
    else:
        left_right = forward_back = 0

        # Reset PID
        prev_error = 0
        integral = 0

        # Flag
        inView = False

    return left_right, forward_back, inView

def navigate_to(current_pos, target_pos, yaw, threshold=10, speed_limit=30):
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = np.hypot(dx, dy)

    if distance < threshold:
        return 0, 0, True

    # Direction vector
    direction_x = dx / distance
    direction_y = dy / distance

    # Convert to drone-relative frame
    rel_x = direction_x * np.cos(-yaw) - direction_y * np.sin(-yaw)
    rel_y = direction_x * np.sin(-yaw) + direction_y * np.cos(-yaw)

    forward_back = int(rel_y * min(speed_limit, distance))
    left_right = int(rel_x * min(speed_limit, distance))

    return left_right, forward_back, False

def align_target(target, telloCentre, dt):
    global prev_error_x, prev_error_y, integral_x, integral_y, error_buffer

    # Default commands
    left_right = 0
    forward_back = 0
    aligned = False

    # Extract coordinates
    target_x, target_y = target[0]
    mid_x, mid_y = telloCentre

    # Calculate errors
    error_x = target_x - mid_x  # → strafe
    error_y = mid_y - target_y  # → move forward/back (flipped for image frame)

    # PID Constants — for slow, smooth motion
    Kp = 0.1
    Ki = 0.01
    Kd = 0.03

    # Integral
    integral_x += error_x * dt
    integral_y += error_y * dt

    # Derivative
    derivative_x = (error_x - prev_error_x) / dt if dt > 0 else 0
    derivative_y = (error_y - prev_error_y) / dt if dt > 0 else 0

    # PID Output
    left_right = int(Kp * error_x + Ki * integral_x + Kd * derivative_x)
    forward_back = int(Kp * error_y + Ki * integral_y + Kd * derivative_y)

    # Clamp to safe range
    left_right = max(-20, min(20, left_right))
    forward_back = max(-20, min(20, forward_back))

    # Update previous error
    prev_error_x = error_x
    prev_error_y = error_y

    print(f"error: {error_x}, {error_y}")
    if (abs(error_x) <= 40) and (abs(error_y) <= 40):
        aligned = True

    # Apply minimum RC output threshold if error is still meaningful
    if abs(error_x) > 35 and abs(left_right) < 10:
        left_right = int(10 * np.sign(left_right))

    if abs(error_y) > 35 and abs(forward_back) < 10:
        forward_back = int(10 * np.sign(forward_back))

    return left_right, forward_back, aligned


def setHeight(targetheight, droneheight):
    # Set limits
    lower_bound = targetheight - 10
    upper_bound = targetheight + 10

    # up_down velocity
    if droneheight < lower_bound:
        return 15
    elif droneheight > upper_bound:
        return -15
    else:
        return 0



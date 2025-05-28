############### Libraries ###############
import time
import cv2
import numpy as np


############### Global Variables ###############
############### PID ###############
# Navigation
prev_error = 0
integral = 0

# Centering
prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0


############### Error Handling and Buffers ###############
# Long term error
error_buffer = [] 
ALIGNMENT_WINDOW = 30

# Gap locking
locked_centroids = None
LOCK_BUFFER = 0


############### Chase Control ###############
def chase(centroids, telloC):
    """
        Chase the largest object using yaw, left_right, and down_up control.
        Raw demonstration of PID control.
    """

    # Get centroid (largest, already ordered by size)
    if len(centroids) >= 1:
        target = centroids[0]
    else:
        return 0,0,0,0
    
    # Intialise commands
    back_forward = 0 
    down_up = 0
    left_right = 0
    yawleft_right = 0

    # Chase
    if centroids:
        # Reference
        target_x = target[0]
        target_y = target[1]
        current_x = telloC[0]
        current_y = telloC[1]

        # Errors
        error_x = target_x - current_x
        error_y = target_y - current_y

        # PID constants
        Kp_yaw = 0.1
        Kp_lr = 0.1
        Kp_ud = 0.1

        # Proportional control
        yawleft_right = int(Kp_yaw * error_x)
        left_right = int(Kp_lr * error_x)
        down_up = int(Kp_ud * (-error_y))

        # Clamp velcocities to commands
        yawleft_right = max(-100, min(100, yawleft_right))
        left_right = max(-100, min(100, left_right))
        down_up = max(-100, min(100, down_up))

    return left_right, back_forward, down_up, -yawleft_right

############### Pole Naviation Control ###############
def navigate_through_poles(centroids, telloCentre, dt):
    """ Navigate through poles. Thread through gaps or avoid single poles. """


    ############### Initialise ###############
    global prev_error, integral
    global locked_centroids, LOCK_BUFFER

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
        if LOCK_BUFFER == 0:
            locked_centroids = (centroids[0], centroids[1])
            LOCK_BUFFER = 5
        else:
            LOCK_BUFFER -= 1

        # Get error
        c1, c2 = locked_centroids
        target_x = (c1[0] + c2[0]) / 2
        error = target_x - mid_x
        sign = np.sign(error)

        # PID control
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        prev_error = error
        output = Kp * error + Ki * integral + Kd * derivative

        # Apply PID output, clamp to limits
        left_right = int(np.clip(sign * output, -40, 40))

        # Flag
        inView = True


    ############### Single Pole Avoidance ###############
    elif len(centroids) == 1:
        # Slow sleed
        forward_back = 10
        
        # Get error
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
        # Fix motion
        left_right = forward_back = 0

        # Reset PID
        prev_error = 0
        integral = 0

        # Flag
        inView = False

    return left_right, forward_back, inView


############### Waypoint Naviation Control ###############
def navigate_to(current_pos, target_pos, yaw, threshold=10, speed_limit=30):
    """ Navigate to coordinates using map as reference. """


    ############### Initialise ###############
    # Relative displacement
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = np.hypot(dx, dy)

    # Threshold of drone to waypoint coordinate
    if distance < threshold:
        return 0, 0, True

    # Direction vector, normalise
    direction_x = dx / distance
    direction_y = dy / distance

    
    ############### Required Action ###############
    # Convert to drone-relative frame. Heading acounted for
    rel_x = direction_x * np.cos(-yaw) - direction_y * np.sin(-yaw)
    rel_y = direction_x * np.sin(-yaw) + direction_y * np.cos(-yaw)

    # Controls
    forward_back = int(rel_y * min(speed_limit, distance))
    left_right = int(rel_x * min(speed_limit, distance))

    return left_right, forward_back, False


############### Align Target/LZ ###############
def align_target(target, telloCentre, dt):
    """ Align bottom camera to centroid. Target or LZ. """

    
    ############### Intialise ###############
    # Global variables
    global prev_error_x, prev_error_y, integral_x, integral_y, error_buffer

    # Defaults
    left_right = 0
    forward_back = 0
    aligned = False

    # Get centroids
    target_x, target_y = target[0]
    mid_x, mid_y = telloCentre


    ############### PID ###############
    # Get errors
    error_x = target_x - mid_x  # left/right
    error_y = mid_y - target_y  # forward/back (flipped for image frame)

    # PID Constants
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

    # Clamp speed
    left_right = max(-20, min(20, left_right))
    forward_back = max(-20, min(20, forward_back))

    # Update previous error
    prev_error_x = error_x
    prev_error_y = error_y


    ############### Alignment ###############
    print(f"error: {error_x}, {error_y}")
    if (abs(error_x) <= 40) and (abs(error_y) <= 40):
        aligned = True

    # Minimum control to actually move drone
    if abs(error_x) > 35 and abs(left_right) < 10:
        left_right = int(10 * np.sign(left_right))

    if abs(error_y) > 35 and abs(forward_back) < 10:
        forward_back = int(10 * np.sign(forward_back))

    return left_right, forward_back, aligned


############### Set Height ###############
def setHeight(targetheight, droneheight):
    """ Get drone to be at specified height, within bounds """

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


############### Attempted: Deadband Control ###############
def apply_deadband_decay(velocity, threshold=10, decay=0.5):
    """ Deadband decay based on small inputs not registering. """

    # Tune outside of function
    if abs(velocity) < threshold:
        return int(velocity * decay)
    
    return velocity



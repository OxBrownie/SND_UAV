############### Libraries ###############
import time
import cv2


############### PID Control ###############
prev_error = 0
integral = 0

def pidYaw(target_x, current_x, dt):
    """ Improved PID controller for Tello yaw control """
    global prev_error, integral

    # PID Gains (Tune these!)
    Kp = 0.3   # Reduce P gain to prevent overshooting
    Ki = 0.005  # Small I gain to prevent drifting
    Kd = 0.3   # Reduce D gain to smooth movements

    # Calculate error (difference between target and current position)
    error = target_x - current_x

    # Dead zone: If error is small, stop movement
    DEAD_ZONE = 50  # Pixels threshold
    if abs(error) < DEAD_ZONE:
        return 0  # No movement needed

    # Proportional term
    P = Kp * error

    # Integral term (accumulates error over time)
    integral += error * dt
    integral = max(-50, min(50, integral))  # Limit integral wind-up
    I = Ki * integral

    # Derivative term (change in error)
    D = Kd * (error - prev_error) / dt if dt > 0 else 0

    # Compute PID output
    yaw_velocity = int(P + I + D)

    # Limit yaw velocity to Tello's range (-100 to 100)
    yaw_velocity = max(-50, min(50, yaw_velocity))  # Reduce max yaw speed for smoother control

    # Update previous error
    prev_error = error

    return yaw_velocity

def pidHeight(target_y, current_y, dt):
    """ Improved PID controller for Tello yaw control """
    global prev_error, integral

    # PID Gains (Tune these!)
    Kp = 0.3   # Reduce P gain to prevent overshooting
    Ki = 0.005  # Small I gain to prevent drifting
    Kd = 0.3   # Reduce D gain to smooth movements

    # Calculate error (difference between target and current position)
    error = target_y - current_y

    # Dead zone: If error is small, stop movement
    DEAD_ZONE = 50  # Pixels threshold
    if abs(error) < DEAD_ZONE:
        return 0  # No movement needed

    # Proportional term
    P = Kp * error

    # Integral term (accumulates error over time)
    integral += error * dt
    integral = max(-50, min(50, integral))  # Limit integral wind-up
    I = Ki * integral

    # Derivative term (change in error)
    D = Kd * (error - prev_error) / dt if dt > 0 else 0

    # Compute PID output
    height_velocity = int(P + I + D)

    # Limit yaw velocity to Tello's range (-100 to 100)
    height_velocity = max(-50, min(50, height_velocity))  # Reduce max yaw speed for smoother control

    # Update previous error
    prev_error = error

    return height_velocity

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
  
def navigate_through_poles(centroids, telloC):
    """Calculate RC control inputs to navigate between poles while moving forward."""

    # Default velocities
    forward_backward_velocity = 13  # Constant forward speed
    up_down_velocity = 0  # No vertical movement
    left_right_velocity = 0  # Strafing left/right
    yaw_velocity = 0  # Default no rotation

    # Proportional control for lateral movement (left/right) and yaw
    Kp_yaw = 0.15  # Tuning gain for yaw (rotation)
    Kp_lr = 0.1    # Tuning gain for left/right movement (lateral)
    Kp_up = 0

    # Center of the frame (drone should aim here)
    mid_x = telloC[0]  
    mid_y = telloC[1]

    centroids.sort(key=lambda c: c[0]) # Sort left to right

    merge_threshold = 50  # Adjust as needed
    if len(centroids) > 1:
        filtered = [centroids[0]]

        for c in centroids[1:]:
            if abs(c[0] - filtered[-1][0]) > merge_threshold:
                filtered.append(c)

        centroids = filtered

    # More than 1 pole
    if len(centroids) >= 2:
        # Get the middle point between the first two centroids (closest poles)
        target_x = (centroids[0][0] + centroids[1][0]) // 2
        target_y = (centroids[0][1] + centroids[1][1]) // 2
        
        # Calculate error 
        error_x = target_x - mid_x    # (+)/(-) = right/left 
        error_y = target_y - mid_y  # (+)/(-) = down/up 

        # Calculate yaw velocity (turn towards the poles)
        forward_backward_velocity = 15
        yaw_velocity = int(Kp_yaw * error_x)
        left_right_velocity = int(Kp_lr * error_x)
        up_down_velocity = int(Kp_up * error_y)  # Negative to go up when object is higher

    elif len(centroids) == 1:
        target_x = centroids[0][0]
        target_y = centroids[0][1]

        # Calculate error (offset from center)
        error_x = target_x - mid_x
        error_y = target_y - mid_y

        # Calculate yaw velocity (turn towards the poles)
        safe_error_x = error_x if abs(error_x) > 1 else 1 if error_x >= 0 else -1
        # yaw_velocity = int(Kp_yaw * error)
        forward_backward_velocity = 13
        left_right_velocity = int(Kp_lr * safe_error_x)
        up_down_velocity = int(Kp_up * (error_y))  # Negative to go up when object is higher

    # Limit yaw and lateral velocities to Tello's range (-100 to 100)
    yaw_velocity = max(-100, min(100, yaw_velocity))
    left_right_velocity = max(-100, min(100, left_right_velocity))
    up_down_velocity = max(-100, min(100, up_down_velocity))

    # Cap speeds
    if left_right_velocity > 10:
        left_right_velocity = 10
    elif left_right_velocity < -10:
        left_right_velocity = -10
    

    # yaw_velocity = int(yaw_velocity * (forward_backward_velocity / 20))  # Scale by forward speed
    # left_right_velocity = int(left_right_velocity * (forward_backward_velocity / 20))  # Scale by forward speed

    return left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity



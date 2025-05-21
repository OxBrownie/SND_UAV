############### Libraries ###############
import cv2
import numpy as np


class Processing:
    ############### Initialise ###############
    def __init__(self, window_name="HSV Threshold", mode=0):
        self.window_name = window_name

        # HSV range values
        self.low_H, self.high_H = 0, 360//2
        self.low_S, self.high_S = 0, 255
        self.low_V, self.high_V = 0, 255

        # Create window and trackbars
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("Low H", self.window_name, self.low_H, 179, self.update_low_H)
        cv2.createTrackbar("High H", self.window_name, self.high_H, 179, self.update_high_H)
        cv2.createTrackbar("Low S", self.window_name, self.low_S, 255, self.update_low_S)
        cv2.createTrackbar("High S", self.window_name, self.high_S, 255, self.update_high_S)
        cv2.createTrackbar("Low V", self.window_name, self.low_V, 255, self.update_low_V)
        cv2.createTrackbar("High V", self.window_name, self.high_V, 255, self.update_high_V)

        # Prests
        cv2.setMouseCallback(window_name, self.set_color_preset)
        self.color_presets = {
            "Red": {"low_H": 0, "high_H": 10, "low_S": 100, "high_S": 255, "low_V": 100, "high_V": 255},
            "Blue": {"low_H": 100, "high_H": 130, "low_S": 50, "high_S": 255, "low_V": 50, "high_V": 255},
            "Pole": {"low_H": 0, "high_H": 97, "low_S": 0, "high_S": 47, "low_V": 175, "high_V": 244},
            "Black": {"low_H": 0, "high_H": 179, "low_S": 0, "high_S": 71, "low_V": 0, "high_V": 94}
        }

        self.polePrest = {
            "low_H": 0, "high_H": 97, "low_S": 0, "high_S": 47, "low_V": 175, "high_V": 244
        }

        # Starting default at pole
        if mode != 0:
            self.poleHSV()


    ############### HSV Filter ###############
    # Slider callbacks
    def update_low_H(self, val):
        self.low_H = min(val, self.high_H - 1)
        cv2.setTrackbarPos("Low H", self.window_name, self.low_H)

    def update_high_H(self, val):
        self.high_H = max(val, self.low_H + 1)
        cv2.setTrackbarPos("High H", self.window_name, self.high_H)

    def update_low_S(self, val):
        self.low_S = min(val, self.high_S - 1)
        cv2.setTrackbarPos("Low S", self.window_name, self.low_S)

    def update_high_S(self, val):
        self.high_S = max(val, self.low_S + 1)
        cv2.setTrackbarPos("High S", self.window_name, self.high_S)

    def update_low_V(self, val):
        self.low_V = min(val, self.high_V - 1)
        cv2.setTrackbarPos("Low V", self.window_name, self.low_V)

    def update_high_V(self, val):
        self.high_V = max(val, self.low_V + 1)
        cv2.setTrackbarPos("High V", self.window_name, self.high_V)

    def printHSV(self):
        print(f"Low: {(self.low_H, self.low_S, self.low_V)}")
        print(f"High: {(self.high_H, self.high_S, self.high_V)}")

    # Apply HSV Filter
    def applyHSV(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        threshold = cv2.inRange(hsv, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))
        return threshold
    
    # Apply pole HSV defaults
    def poleHSV(self):
        """ Apply HSV filtering to detect white colors (poles) """

        # Convert to HSV color space
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Set pole preset colours
        self.low_H = self.polePrest["low_H"]
        self.high_H = self.polePrest["high_H"]
        self.low_S = self.polePrest["low_S"]
        self.high_S = self.polePrest["high_S"]
        self.low_V = self.polePrest["low_V"]
        self.high_V = self.polePrest["high_V"]

        # Update trackbars to reflect the preset values
        cv2.setTrackbarPos("Low H", self.window_name, self.low_H)
        cv2.setTrackbarPos("High H", self.window_name, self.high_H)
        cv2.setTrackbarPos("Low S", self.window_name, self.low_S)
        cv2.setTrackbarPos("High S", self.window_name, self.high_S)
        cv2.setTrackbarPos("Low V", self.window_name, self.low_V)
        cv2.setTrackbarPos("High V", self.window_name, self.high_V)

        # # Threshold the image to extract only white colors
        # threshold = cv2.inRange(hsv, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))

        # return threshold


    ############### Preset Colours ###############
    # Button
    def draw_buttons(self, frame):
        # Red button
        cv2.rectangle(frame, (10, 10), (150, 60), (0, 0, 0), -1)  # Red color
        cv2.putText(frame, 'Red', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Blue button
        cv2.rectangle(frame, (160, 10), (300, 60), (0, 0, 0), -1)  # Blue color
        cv2.putText(frame, 'Blue', (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Silver button
        cv2.rectangle(frame, (310, 10), (450, 60), (0, 0, 0), -1)  # Silver color
        cv2.putText(frame, 'Pole', (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Black button
        cv2.rectangle(frame, (460, 10), (600, 60), (0, 0, 0), -1)  # Black color
        cv2.putText(frame, 'Black', (490, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Set colour
    def set_color_preset(self, event, x, y, flags, param):
        
        # Check if the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Button regions (clickable areas)
            if 10 < x < 150 and 10 < y < 60:  # Red button
                preset = self.color_presets["Red"]
            elif 160 < x < 300 and 10 < y < 60:  # Blue button
                preset = self.color_presets["Blue"]
            elif 310 < x < 450 and 10 < y < 60:  # Silver button
                preset = self.color_presets["Pole"]
            elif 460 < x < 600 and 10 < y < 60:  # Black button
                preset = self.color_presets["Black"]
            else:
                return  # If clicked outside the button area, do nothing
            
            # Set the HSV values based on the preset
            self.low_H = preset["low_H"]
            self.high_H = preset["high_H"]
            self.low_S = preset["low_S"]
            self.high_S = preset["high_S"]
            self.low_V = preset["low_V"]
            self.high_V = preset["high_V"]

            # Update trackbars to reflect the preset values
            cv2.setTrackbarPos("Low H", self.window_name, self.low_H)
            cv2.setTrackbarPos("High H", self.window_name, self.high_H)
            cv2.setTrackbarPos("Low S", self.window_name, self.low_S)
            cv2.setTrackbarPos("High S", self.window_name, self.high_S)
            cv2.setTrackbarPos("Low V", self.window_name, self.low_V)
            cv2.setTrackbarPos("High V", self.window_name, self.high_V)
                
            print(f"Preset {list(self.color_presets.keys())[list(self.color_presets.values()).index(preset)]} selected")


    ############### Object detect ###############
    # Colour vision
    def getMask(self, threshold):
        # Initial Gaussian blur
        blurred = cv2.GaussianBlur(threshold, (15, 15), 0)

        # Binary thresholding to remove low-intensity noise
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)    # Remove small noise

        # Optional: Remove small blobs by area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # You can tweak this threshold
                cv2.drawContours(mask_clean, [cnt], -1, 255, -1)

        return mask_clean

    def getPoleMask(self, threshold):
        # Morphology
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove small noise

        # Optional: smooth & erode a bit to avoid blob fusion
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    # Focus detect
    def objectDetect(self, threshold, frame):
        """ Detect the largest shape from the frame and display its centroid """

        # Basic colour processing
        mask = self.getMask(threshold)

        # Contouring
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # Sort contours by area to focus on the largest contour (most prominent shape)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Check if there are any contours (Only want one)
        if contours:
            # Focus only on the largest contour (the first one in the sorted list)
            largest_contour = contours[0]

            # Approximate the contour to reduce the number of vertices and smooth the outline
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(approx)

            # Calculate the moments to get the centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                # Calculate centroid
                Cx = int(M["m10"] / M["m00"])
                Cy = int(M["m01"] / M["m00"])
            else:
                Cx, Cy = 0, 0  # Default to (0, 0) if division by zero

            # Classify shape based on the number of vertices
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif len(approx) > 6:
                shape = "Circle"
            else:
                shape = "Unknown"

            # Annotate the largest shape on the frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)  # Draw the contour
            # cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw the centroid (red circle) and label the coordinates
            cv2.circle(frame, (Cx, Cy), 5, (0, 0, 255), -1)  # Draw a red circle at the centroid
            cv2.putText(frame, f"Centroid: ({Cx}, {Cy})", (Cx + 10, Cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Return the frame, the largest contour approximation, and the centroid
            return frame, (Cx, Cy)

        # Return the original frame, 
        return frame, []
    
    # Pole detect
    def poleDetect(self, threshold, frame):
        """Detect and label contours in the thresholded frame."""

        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200  # Adjust this value as needed
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Sort contours by area to focus on the largest contour (closest?)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)


        # Centroids
        centroids = []
        counter = 1
        for contour in contours:
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            aspect_ratio = h / float(w)

            # Determine "poles"
            if aspect_ratio > 2:
                # if (y > frame.shape[0] * 0.9) or (y < frame.shape[0] * 0.2):  # very low on frame
                #     continue

                # Draw conour
                if counter < 3:
                    cv2.drawContours(frame, [contour], -1, (128, 128, 128), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.drawContours(frame, [contour], -1, (128, 128, 128), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                counter += 1
                rectangle = True

            else:
                # Draw the contour outline in green
                cv2.drawContours(frame, [contour], -1, (192, 192, 192), 2)
                rectangle = False

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                Cx = int(M["m10"] / M["m00"])
                Cy = int(M["m01"] / M["m00"])

                if rectangle:
                    centroids.append((Cx, Cy))
                    # Draw the centroid and label the contour
                    cv2.circle(frame, (Cx, Cy), 4, (255, 0, 0), -1)
                    cv2.putText(frame, f"({Cx}, {Cy})", (Cx + 10, Cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    pass
                    # # Draw the centroid and label the contour
                    # cv2.circle(frame, (Cx, Cy), 4, (0, 0, 255), -1)
                    # cv2.putText(frame, f"({Cx}, {Cy})", (Cx + 10, Cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame, centroids

    def YOLODetectPoles(self, model, results, frame):
        ############### Initialise ###############
        centroids = []
        centroid_boxes = []

        # Extract boxes and get centroids
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = abs((x2 - x1) * (y2 - y1))
                centroid_boxes.append(((cx, cy), (x1, y1, x2, y2), conf, label, area))


        ############### Get Gap ###############
        # Sort boxes left to right by x
        centroid_boxes.sort(key=lambda item: item[0][0])

        # Determine widest gap between adjacent poles
        max_gap = -1
        max_pair = (None, None)
        for i in range(len(centroid_boxes) - 1):
            x1 = centroid_boxes[i][0][0]
            x2 = centroid_boxes[i + 1][0][0]
            gap = abs(x2 - x1)
            if gap > max_gap:
                max_gap = gap
                max_pair = (i, i + 1)

        
        ############### Bounding Boxes ###############
        for i, (centroid, (x1, y1, x2, y2), conf, label, area) in enumerate(centroid_boxes):
            # Colour
            if i in max_pair:
                colour = (255, 0, 0) # Blue
            else:
                colour = (0, 255, 0) # Green

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{label}, {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Draw centroid
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
            centroids.append(centroid)

        
        ############### Visualsie Targets ###############
        if len(centroids) >= 2 and max_pair[0] is not None:
            c1 = centroids[max_pair[0]]
            c2 = centroids[max_pair[1]]
            target_x = (c1[0] + c2[0]) // 2
            target_y = (c1[1] + c2[1]) // 2
            cv2.circle(frame, (target_x, target_y), 6, (255, 0, 0), -1)
            cv2.putText(frame, "TARGET", (target_x + 10, target_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elif len(centroids) == 1:
            c = centroids[0]
            cv2.circle(frame, c, 6, (255, 0, 0), -1)
            cv2.putText(frame, "AVOID", (c[0] + 10, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame, centroids

    def YOLODetectTarget(self, model, results, frame):
        ############### Initialise ###############
        target_boxes = []
        targets = []

        # Extract boxes and get centroids
        for r in results:
            for box in r.boxes:
                label = model.names[cls]
                if label == 'Target':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    area = abs((x2 - x1) * (y2 - y1))
                    target_boxes.append(((cx, cy), (x1, y1, x2, y2), conf, label, area))

        
        ############### Bounding Boxes ###############
        for i, (centroid, (x1, y1, x2, y2), conf, label, area) in enumerate(target_boxes):
            # Colour
            colour = (255, 0, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{label}, {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Draw centroid
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
            targets.append(centroid)

        return frame, targets

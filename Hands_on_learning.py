# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:19:41 2019

@author: hp1
"""

import cv2
import math
import os

#To define all the directories that will be used in the code.
base_dir= os.path.dirname(os.path.abspath(__file__))

class Hand:
    def __init__(self,binary,masked,raw,frame):
        self.binary = binary
        self.masked = masked
        self.raw = raw
        self.frame = frame
        self.contours = []
        self.outline = self.draw_outline()
        self.fingertips = self.locate_fingers()
        
    ### To draw an outline around the image of the hand that was detected using the function detect_hand
    def draw_outline(self, min_area=10000, color=(0,255,0), thickness=2):
        
        # findContour is used to get a curve joining all the continuous points (along the boundary), having the same color or intensity. It gives the Contours and Hierarchy as output. We are using only the contours.
        # RETR_TREE retrieves all the contours and creates a full family hierarchy list. Ref https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        # CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour to save memory.
        contours, _ = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        palm_area = 0
        flag = None
        cnt = None
        for (index, contour) in enumerate(contours):        #Enumerate() method adds a counter to an iterable. Ref https://www.geeksforgeeks.org/enumerate-in-python/
            area = cv2.contourArea(contour)                 #Get the area of the palm in the image.
            if area > palm_area:                            #
                palm_area = area
                flag = index
        if flag is not None and palm_area > min_area:
            cnt = contours[flag]
            self.contours = cnt
            duplicate_frame = self.frame.copy()
            cv2.drawContours(duplicate_frame, [cnt], 0, color, thickness)
            return duplicate_frame
        else:
            return self.frame
        
    def locate_fingers(self, filter_value=50):
        cnt = self.contours
        if len(cnt) == 0:
            return cnt
        else:
            points = []
            #Generally speaking, convex curves are the curves which are always bulged out, or at-least flat. 
            #And if it is bulged inside, it is called convexity defects. Deviation of the object from the convex hull is considered a convexity defect
            hull = cv2.convexHull(cnt, returnPoints=False)              #Checks a curve for convexity defects and corrects it. Ref https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
            defects = cv2.convexityDefects(cnt, hull)                   #Returns an array where each row contains these values - [ start point, end point, farthest point, approximate distance to farthest point ].
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                end = tuple(cnt[e][0])
                points.append(end)
            filtered = self.filter_points(points, filter_value)
            
            filtered.sort(key=lambda point: point[1])
            return [pt for idx, pt in zip(range(5), filtered)]
        
    def filter_points(self, points, filter_value):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if points[i] and points[j] and self.dist(points[i], points[j]) < filter_value:
                    points[j] = None
        filtered = []
        for point in points:
            if point is not None:
                filtered.append(point)
        return filtered
    
    def dist(self, a, b):
        return math.sqrt((a[0] - b[0])**2+(a[1] - b[1])**2)
    
    def get_center_of_mass(self):
        if len(self.contours) == 0:
            return None
        else:
            M = cv2.moments(self.contours)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            return (center_X, center_Y)

def locate_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject(
        [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cv2.filter2D(object_segment, -1, disc, object_segment)

    _, segment_thresh = cv2.threshold(
        object_segment, 70, 255, cv2.THRESH_BINARY)

    # apply some image operations to enhance image
    kernel = None
    eroded = cv2.erode(segment_thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # masking
    masked = cv2.bitwise_and(frame, frame, mask=closing)

    return closing, masked, segment_thresh

### A function to take a picture of the user's hand that will be used for the purpose of detection ###
def capture_histogram(source=0):
    cap = cv2.VideoCapture(source)                                                      # Start webcam
    while True:
        _, frame = cap.read()                                                           # Read image seen on webcam
        frame = cv2.flip(frame, 1)                                                      # To flip the image horizontally about the Y axis
        frame = cv2.resize(frame, (1000, 600))                                          # Resize the image to fit requirements

        font = cv2.FONT_HERSHEY_SIMPLEX                                                 # Assign a font for the text to be shown on screen.
        cv2.putText(frame, "Place region of the hand inside box and press `A`",
                    (5, 50), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)                # Ref https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        cv2.rectangle(frame, (500, 100), (700, 300), (105, 105, 105), 2)                # Draw a rectangle using the points given as input.
        box = frame[100:300, 500:700]                                                   # Extract of only the area with the hand.

        cv2.imshow("Capture Histogram", frame)                                          # Display the image with the rectangle and text.
        key = cv2.waitKey(10)
        if key == 97:                                                                   # When the user presses 'ESC', save the latest captured image as the hand.
            object_color = box                                                          
            cv2.destroyAllWindows()
            break
        if key == 27:                                                                   # When the user presses 'numpad 1', then the hardware and software is released and the program shuts.
            cv2.destroyAllWindows()
            cap.release()
            break
        
    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)                    # Convert from BGR to HSV color-space (Hue Saturation and Value), where it is easier to extract a colored object.
    object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
                               [12, 15], [0, 180, 0, 256])                              # Calculate the histogram of one or more arrays.
    
    # Normalizes the norm or value range of an array.
    # NORM_MINMAX normalizes the image to set the min value of dist as alpha and max value of dist as beta
    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    cap.release()
    return object_hist

def detect_hand(frame, hist):
    detected_hand, masked, raw = locate_object(frame, hist)
    return Hand(detected_hand, masked, raw, frame)


# getting video feed from webcam
cap = cv2.VideoCapture(0)

# capture the hand histogram by placing your hand in the box shown and press 'A' to confirm
# source is set to inbuilt webcam by default. Pass source=1 to use an external camera.
hist = capture_histogram(source=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect the hand
    hand = detect_hand(frame, hist)

    # to get the outline of the hand. Min area of the hand to be detected = 10000 by default
    custom_outline = hand.draw_outline(min_area=10000, color=(0, 255, 255), thickness=2)

    # to get a quick outline of the hand
    quick_outline = hand.outline

    # draw fingertips on the outline of the hand, with radius 5 and color red,
    # filled in.
    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)

    # to get the centre of mass of the hand
    com = hand.get_center_of_mass()
    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)

    cv2.imshow("Handy", quick_outline)

    # display the unprocessed, segmented hand
    # cv2.imshow("Handy", hand.masked)

    # display the binary version of the hand
    # cv2.imshow("Handy", hand.binary)

    # Press 'q' to exit
    key = cv2.waitKey(1)
    if key==ord("q") or key==ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()
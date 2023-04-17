import cv2
import hand_tracking as htm
import argparse
import numpy as np
import json
import imutils 
import vgamepad as vg
import itertools
import math
from constants import ControllerConstants


def find_color(frame, points, thresh):
    mask = cv2.inRange(frame, points[0], points[1])#create mask with boundaries 
    cnts = cv2.findContours(mask, cv2.RETR_TREE, 
                           cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv2.contourArea(c) 
        if area > thresh:       
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00']) # calculate X position
            cy = int(M['m01'] / M['m00']) # calculate Y position
            return c, cx, cy
    return None
def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--bgr_range_path', type=str, default='color-ranges\color-values-gopro-table.json', help='JSON file to load max and min BGR values for each color')
    parser.add_argument('--draw_buttons_interval', type=int, default=10, help='Reset button contours every X seconds')
    parser.add_argument('--difficulty', type=str, default='easy', help='difficulty level. maps middle 3 right fingers to green, red, and yellow respectively')
    parser.add_argument('--distance', type=float, default=20.0, help='distance finger has to be to box')
    parser.add_argument('--use_center', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--area_threshold', type=int, default=1000)
    args = parser.parse_args()
    return args
def main():
    gamepad = vg.VX360Gamepad()
    args = _parse_args() 
    cap=cv2.VideoCapture(0)
    DIFFICULTY = args.difficulty
    DISTANCE = args.distance
    INTERVAL = args.draw_buttons_interval
    AREA_THRESH = args.area_threshold
    VERBOSE = args.verbose
    USE_CENTER = args.use_center
    # I have defined lower and upper boundaries for each color for my camera
    # Strongly recommended finding for your own camera.
    with open(args.bgr_range_path) as file:
        data = json.load(file)
    colors = {}
    for color in data:
        vals = list(data[color].values())
        colors[color] = (np.array(vals[0:3], dtype=np.uint8), np.array(vals[3:], dtype=np.uint8))

    detector = htm.handDetector(detectionCon=0.75)
    prev_colors_pressed = set()
    count = 0 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if VERBOSE:
        print("Active colors: \n", colors)
        print("Capture FPS: ", fps)
    while True:
        _,frame=cap.read()
        frame = detector.findHands(frame, draw=True )
        lm_list=detector.findPosition(frame,draw=False)
        x_y_list = []
        if(len(lm_list) > 1):
            # compare x y of finger tip to corresponding knuckle
            if lm_list[8][2]>lm_list[5][2]: # pointer
                x_y_list.append((lm_list[8][1], lm_list[8][2]))
            if lm_list[12][2]>lm_list[9][2]: # middle
                x_y_list.append((lm_list[12][1], lm_list[12][2]))
            if lm_list[16][2]>lm_list[13][2]: # index
                x_y_list.append((lm_list[16][1], lm_list[16][2]))
            if lm_list[20][2]>lm_list[17][2]: # pinky
                x_y_list.append((lm_list[20][1], lm_list[20][2]))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
        if count % INTERVAL == 0:
            contour_buffer = []
            for name, clr in colors.items(): # for each color in colors
                found_color = find_color(hsv_frame, clr, AREA_THRESH)
                if found_color is not None:
                    c, cx, cy = found_color
                    contour_buffer.append((name, (c, cx, cy)))
                    if USE_CENTER:
                        cv2.circle(frame, (cx, cy), 2, ControllerConstants.CYAN, 2)
                    else:
                        cv2.drawContours(frame, [c], -1, ControllerConstants.CYAN, 3) #draw contours
        else:
            if USE_CENTER:
                for contour in contour_buffer:
                    cv2.circle(frame, (contour[1][1], contour[1][2]), 2, ControllerConstants.WHITE, 2)
            else:
                cv2.drawContours(frame, [contour[1][0] for contour in contour_buffer], -1, ControllerConstants.WHITE, 3)
        cur_colors_pressed = set()
        if len(x_y_list) > 0 and len(contour_buffer) > 0:  # call find_color function above
            for x_y, contour in itertools.product(x_y_list, contour_buffer):
                if USE_CENTER:
                    c, cx, cy = contour[1]
                    x, y = x_y
                    result = math.hypot(cx - x, cy - y)
                else:
                    result = cv2.pointPolygonTest(contour, x_y, True)
                if abs(result) < DISTANCE:
                    cur_colors_pressed.add(contour[0])
                    if VERBOSE and USE_CENTER:
                        print(contour[0], abs(result))
        for color in cur_colors_pressed:
            gamepad.press_button(ControllerConstants.INPUTS[color])
        gamepad.update()
        colors_to_release = prev_colors_pressed.difference(cur_colors_pressed)
        for color in colors_to_release:
            gamepad.release_button(ControllerConstants.INPUTS[color])
        gamepad.update()
        print(cur_colors_pressed, prev_colors_pressed)
        prev_colors_pressed = cur_colors_pressed
        count += 1
        cv2.imshow("image",frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
if __name__ == "__main__":
    main()
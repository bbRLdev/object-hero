import cv2
import hand_tracking as htm
import time
import argparse
import numpy as np
import json
import imutils 


def find_color(frame, points):
    mask = cv2.inRange(frame, points[0], points[1])#create mask with boundaries 
    cnts = cv2.findContours(mask, cv2.RETR_TREE, 
                           cv2.CHAIN_APPROX_SIMPLE) # find contours from mask
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv2.contourArea(c) # find how big countour is
        if area > 1000:       # only if countour is big enough, then
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
    # parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    # parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    # parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    # parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    # parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    # parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word embeddings to use')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--feats', type=str, default='UNIGRAM', help='features if using linear model')
    # parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    # parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    # parser.add_argument('--batch_size', type=int, default=1, help='training batch size; 1 by default and you do not need to batch unless you want to')
    args = parser.parse_args()
    return args
def main():    
    args = _parse_args() 
    cap=cv2.VideoCapture(0)
    WHITE = (255,255,255)
    # I have defined lower and upper boundaries for each color for my camera
    # Strongly recommended finding for your own camera.
    with open(args.bgr_range_path) as file:
        data = json.load(file)
    colors = {}
    for color in data:
        vals = list(data[color].values())
        colors[color] = (np.array(vals[0:3], dtype=np.uint8), np.array(vals[3:], dtype=np.uint8))
    print(colors)
    detector = htm.handDetector(detectionCon=0.75)
    while True:
        success,frame=cap.read()
        frame = detector.findHands(frame, draw=True )
        lmList=detector.findPosition(frame,draw=False)
        pointer_x_y = None
        if(len(lmList) > 1 and lmList[8][2]>lmList[6][2]):
            pointer_x_y = (lmList[8][1], lmList[8][2])
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
        # mask =  cv2.inRange(hsv_frame, colors['blue'][0], colors['blue'][1])
        # bitwise = cv2.bitwise_and(mask, motion_frame)
        for name, clr in colors.items(): # for each color in colors
            found_color = find_color(hsv_frame, clr)
            if found_color is not None:  # call find_color function above
                c, cx, cy = found_color
                cv2.drawContours(frame, [c], -1, WHITE, 3) #draw contours
                if(pointer_x_y != None):
                    result = cv2.pointPolygonTest(c, pointer_x_y, True)
                    if abs(result) < 20:
                        print(name) 
        cv2.imshow("image",frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break


if __name__ == "__main__":
    main()
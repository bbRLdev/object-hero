import cv2
import imutils
# Credit: https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
def find_color(frame, range, thresh, mask=None):
    """Returns the largest contour according to the area of the 
    frame with the most pixels of a certain color.

    Args:
        frame (np.array): Frame to get contours from
        range (tuple::<np.array,np.array>): min and max hsv values. 
        to create mask from thresh (float): Area to draw a contour 
        from if it is large enough.
        mask (np.array, optional): Mask of shape (H, W) from cv2.inRange() 
        to find contours on. Masks out individual blocks.
    Returns:
        tuple::<contour, int cx, int cy> or None: tuple with contour, 
        and contour center x and y from cv2.findContours(). Returns
        None if no contour large enough was found.
    """
    #create mask with boundaries 
    if mask is None:
        mask = cv2.inRange(frame, range[0], range[1])
    cnts = cv2.findContours(mask, cv2.RETR_TREE, 
                           cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv2.contourArea(c) 
        if area > thresh:       
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00']) 
            return c, cx, cy
    return None
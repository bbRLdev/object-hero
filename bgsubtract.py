
import cv2
def main():
    # creating object
    fgbg2 = cv2.createBackgroundSubtractorMOG2()
    fgbg3 = cv2.createBackgroundSubtractorKNN()
    
    # capture frames from a camera 
    cap = cv2.VideoCapture(0)
    while(1):
        # read frames
        ret, img = cap.read()
        
        # apply mask for background subtraction
        fgmask2 = fgbg2.apply(img)
        fgmask3 = fgbg3.apply(img)
    
        cv2.imshow('Original', img)
        cv2.imshow('MOG2', fgmask2)
        cv2.imshow('knn', fgmask3)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
  
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import time
 
 
class HandDetector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, track_confidence=0.5):
        """_summary_

        Args:
            mode (bool, optional): _description_. Defaults to False.
            max_hands (int, optional): _description_. Defaults to 2.
            model_complexity (int, optional): _description_. Defaults to 1.
            detection_confidence (float, optional): _description_. Defaults to 0.5.
            track_confidence (float, optional): _description_. Defaults to 0.5.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.model_complexity = model_complexity
 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                        self.detection_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils
 
    def find_hands(self, img, draw_complex=True):
        """This method finds the position of our hands

        Args:
            img (np.array): The frame we wish to find our hands within
            draw_complex (bool, optional): Controls whether mediapipe draws a skeleton of inferred hand position. Defaults to True.

        Returns:
            np.array, list::<tuple::<int, int, int>>: Returns the image, now with a hand drawn on it if draw_complex=True, 
            as well a list of tuples containing the hand part id (finger, palm, knuckle, etc.) and its x, y position.
            To see the ids and which part of the hand they correspond to, refer to 
            https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        positions = []
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_complex:
                    self.mp_draw.draw_landmarks(img, hand,
                                               self.mp_hands.HAND_CONNECTIONS)
                lm_list = []
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h) # translate back to W, H scale coordinates
                    lm_list.append((id, cx, cy))
                positions.append(lm_list)
        return img, positions
 
def main():
    time_start = 0
    time_end = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img, _ = detector.find_hands(img)
 
        time_end = time.time()
        fps = 1 / (time_end - time_start)
        time_start = time_end
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()
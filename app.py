import cv2
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# gets a random fruit from the list of fruits
def get_random_fruit() -> str:
    # Create fruit
    fruits = ["Apple.png", "Banana.png", "BellPepper-orange_Half.png", "BellPepper-orange.png", "Jalepeño_Half.png", "Jalepeño.png", "Orange.png", "Tomato_Half.png", "Tomato.png"]
    return random.choice(fruits)


def main():

    # initializes the camera
    # Capture camera feed
    cap = cv2.VideoCapture(0)
    # check if camera was unable to be openend successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # get random fruit
    fruit = get_random_fruit() # gets a random fruit from the list of fruits
    fruit_image = cv2.imread("/Users/ryan/Desktop/Paloma's Fruit Icons/" + fruit, cv2.IMREAD_UNCHANGED) # reads the fruit image
    fruit_x, fruit_y = 500, 500 # location of where the fruit imag is going to be places on screen

    # check if fruit image was unable to be opened successfully
    if fruit_image is None:
        print("Error: Could not open fruit image.")
        return

    # create HandLandmarker object
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,  # Number of hands to detect
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Main Loop to capture frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # process the frame with HandLandmarker
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB for mediapipe
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        detection_result = detector.process(mp_frame)

        # check for landmarks
        if detection_result and detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            # get the x and y coordinates of the index finger tip
            index_tip = hand_landmarks[8]
            index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

            # check if finger is near the fruit (simple bounding box detection)
            if fruit_x <= index_x  <= fruit_x + fruit_image.shape[1] and \
                fruit_y <= index_y <= fruit_y + fruit_image.shape[0]:
                # simulate "clicking" the fruit
                print("fresh cut!")
                fruit = get_random_fruit()
                fruit_image = cv2.imread("/Users/ryan/Desktop/Paloma's Fruit Icons/" + fruit, cv2.IMREAD_UNCHANGED)

        # Blending between fruit and ROI
        b, g, r, alpha = cv2.split(fruit_image) # seperates fruit images into its color channels (RGB) and alpha channel (transperancy)
        alpha_mask = alpha / 255.0 # controls how transparent the each pixel of the fruit image is
        alpha_mask_inv = 1.0 - alpha_mask

        # define ROI (Region of Interest aka where the fruit image will appear) in the camera frame
        roi = frame[fruit_y:fruit_y + fruit_image.shape[0], fruit_x:fruit_x+fruit_image.shape[1]]

        # Use the alpha_mask to blend the fruit image with the ROI.
        for c in range(3):  # Blend each color channel (RGB)
            roi[:, :, c] = (alpha_mask * fruit_image[:, :, c] + alpha_mask_inv * roi[:, :, c])
        
        # Place the blended result back into the camera feed at the specified location.
        frame[fruit_y:fruit_y + fruit_image.shape[0], fruit_x:fruit_x + fruit_image.shape[1]] = roi

        # Display the camera feed
        cv2.imshow("Camera View", frame)

        # Keyboard Inputs
        # press r to get a new random fruit
        if cv2.waitKey(1) & 0xFF == ord('r'):
            fruit = get_random_fruit()
            fruit_image = cv2.imread("/Users/ryan/Desktop/Paloma's Fruit Icons/" + fruit, cv2.IMREAD_UNCHANGED)

        # Press ESC to exit the program
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    # when user presses 'esc' key then exit the program
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
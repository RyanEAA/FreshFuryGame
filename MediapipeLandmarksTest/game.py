import cv2
import mediapipe as mp
import fruit_utils as fu
import time
import random

# setup up for video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# creates hands object
mp_hands = mp.solutions.hands # module for hand detection and landmarking
hand = mp_hands.Hands(max_num_hands=1) # creates object to process hand landmarks
mp_drawing = mp.solutions.drawing_utils # module that helps to draw landmarks on the frame

# keep track of num_finger_markers index finger positions
num_finger_markers = 7
index_finger_positions = []

# get fruit string
fruit_string = fu.get_random_fruit()
fruit_image =  cv2.imread(fruit_string, cv2.IMREAD_UNCHANGED) # reads the fruit image
#print("fruit_image: ", fruit_image)

# check if image was loaded successfully
if fruit_image is None:
    print("Error: Could not open fruit image.")
    exit()

# fruit spawn and display time
FRUIT_SPAWN_TIME_SECONDS = 1
FRUIT_SCREEN_TIME_SECONDS = 2
last_spawn_time = 0
last_display_time = 0
is_fruit_spawned = False

# get dimensions of fruit image
fruit_h, fruit_w, fruit_c = fruit_image.shape
print("successfully got fruit image shape")
fruit_x, fruit_y = 600, 300 # location of where the fruit image is going to be placed on screen

viable_x_spawn = [fruit_w, 1280 - fruit_w]
viable_y_spawn = [fruit_h, 720 - fruit_h]

fruit_cut_count = 0
while True:
    # captures frame
    success, frame = cap.read() # this data comes in BGR format

    # if frame was captured successfully
    # game can start
    if success:

        # frame handling
        frame = cv2.flip(frame, 1) # flips the frame horizontally
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converts the frame to RGB format

        # draw fruit image at random location on screen
        fruit_top_left = (fruit_x - fruit_w//2, fruit_y - fruit_h//2)
        fruit_bottom_right = (fruit_x + fruit_w//2, fruit_y + fruit_h//2)

        current_time = time.time()

        # fruit spawn every 2 seconds and only 1 fruit on screen at a time
        if current_time - last_spawn_time >= FRUIT_SPAWN_TIME_SECONDS and not is_fruit_spawned:
            last_spawn_time = current_time
            is_fruit_spawned = True
            last_display_time = current_time

            # randomize spawn location
            fruit_x = random.randint(viable_x_spawn[0], viable_x_spawn[1])
            fruit_y = random.randint(viable_y_spawn[0], viable_y_spawn[1])
            print("spawn fruit")

        # if fruit is spawned and it has been less than FRUIT_SCREEN_TIME_SECONDS since it was last displayed
        if is_fruit_spawned and current_time - last_display_time <= FRUIT_SCREEN_TIME_SECONDS:
            # check boundaries
            if (0 <= fruit_x < frame.shape[1] - fruit_w) and (0 <= fruit_y < frame.shape[0] - fruit_h):
                # Define ROI (Region of Interest) in the camera frame
                roi = frame[fruit_y:fruit_y + fruit_h, fruit_x:fruit_x + fruit_w]

                # Ensure the ROI matches the fruit image dimensions
                if roi.shape[:2] == (fruit_h, fruit_w):
                    # Get the alpha channel of the fruit image
                    b, g, r, alpha = cv2.split(fruit_image)
                    alpha_mask = alpha / 255.0
                    alpha_mask_inv = 1.0 - alpha_mask

                    # Blend the fruit image with the ROI
                    for c in range(3):
                        roi[:, :, c] = (alpha_mask * fruit_image[:, :, c] + alpha_mask_inv * roi[:, :, c])

                    # Place the blended ROI back into the frame
                    frame[fruit_y:fruit_y + fruit_h, fruit_x:fruit_x + fruit_w] = roi
            else:
                print("Fruit out of bounds")
                fruit_x, fruit_y = 600, 300
                is_fruit_spawned = False

        elif is_fruit_spawned and current_time - last_display_time > FRUIT_SCREEN_TIME_SECONDS:
            is_fruit_spawned = False
            fruit_string = fu.get_random_fruit() # get new fruit string
            fruit_image = cv2.imread(fruit_string, cv2.IMREAD_UNCHANGED) # gets new fruit images
            fruit_h, fruit_w, fruit_c = fruit_image.shape # get new fruit dimensions


            # updates viable spawn ranges when fruit size changes
            viable_x_spawn = [fruit_w//2, 1280 - fruit_w//2]
            viable_y_spawn = [fruit_h//2, 720 - fruit_h//2]
        
        processsed_frame = hand.process(RGB_frame) # processes RGB frame to get hand landmarks

        ### if hand landmarks are detected
        if processsed_frame.multi_hand_landmarks:
            hand_landmark = processsed_frame.multi_hand_landmarks[0] # gets the first hand landmark
            
            # draws all the landmarks on the hand
            #mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # extract the index finger tip landmark (Landmark #8)
            index_finger_tip = hand_landmark.landmark[8]  # mediapipe index finger tip
            height, width, _ = frame.shape  # Get the dimensions of the frame
            
            # Convert normalized coordinates to pixel coordinates
            x = int(index_finger_tip.x * width)
            y = int(index_finger_tip.y * height)
            
            # appends the index finger tip coordinates to the list
            index_finger_positions.append((x, y))

            # if there's more than num_finger_markers index finger positions, remove the first one
            if len(index_finger_positions) > num_finger_markers:
                index_finger_positions.pop(0)
            

            # draws a blue circle at the index finger tip
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1) # radius = 10
            # draw a line connecting the index finger tip to the previous index finger tip
            for i in range(1, len(index_finger_positions)):
                # get current position
                current_position = index_finger_positions[i]
                # get previous position
                previous_position = index_finger_positions[i - 1]
                cv2.line(frame, previous_position, current_position, (0,255,0), 3 )

            # check if index finger tip is within the fruit image
            if (x >= fruit_x and x <= fruit_x + fruit_w) and (y >= fruit_y and y <= fruit_y + fruit_h):
                print("Fresh cut!")
                fruit_cut_count += 1 # updates fruit cut count

                # get new fruit string
                fruit_string = fu.get_random_fruit()
                # generate new locations
                fruit_x = random.randint(viable_x_spawn[0], viable_x_spawn[1])
                fruit_y = random.randint(viable_y_spawn[0], viable_y_spawn[1])
                # get new fruit image
                fruit_image = cv2.imread(fruit_string, cv2.IMREAD_UNCHANGED)
                fruit_h, fruit_w, fruit_c = fruit_image.shape   


        # adds counter to screen
        counter_text = f"Counter: {fruit_cut_count}"
        cv2.putText(frame, counter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        cv2.imshow("Frame", frame)

        # if user presses 'q' key then exit the program
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

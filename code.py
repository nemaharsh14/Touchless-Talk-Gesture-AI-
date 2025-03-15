import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai

# Configure the generative AI model with your API key
genai.configure(api_key="AIzaSyCPAu4eoksXmsN45YB_dJxELNBmIJTkEr8")
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# Start video capture from the webcam
capture = cv2.VideoCapture(0)

# Initialize the hand detector with specified parameters
hand_detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

previous_position = None
drawing_canvas = None

# Function to save the drawing as an image file
def save_drawing(canvas):
    image = Image.fromarray(canvas)
    image.save("hand_drawing.png")

# Function to send the drawing to an AI model
def analyze_drawing_with_ai(canvas):
    image = Image.fromarray(canvas)
    response = ai_model.generate_content(["Analyze this drawing:", image])
    print(response.text)

# Main loop to process video frames
while True:
    # Read a frame from the webcam
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(frame)
    
    # Detect hands and landmarks in the current frame
    hands, frame = hand_detector.findHands(frame, draw=False, flipType=True)
    if hands:
        # Extract the first detected hand's landmarks
        hand_data = hands[0]
        landmarks_list = hand_data["lmList"]
        fingers_up = hand_detector.fingersUp(hand_data)
        
        # Draw if the index finger is raised
        if fingers_up == [0, 1, 0, 0, 0]:
            current_position = landmarks_list[8][:2]
            if previous_position is not None:
                # Draw on the canvas
                cv2.line(drawing_canvas, tuple(previous_position), tuple(current_position), (0, 0, 255), 5)
            previous_position = current_position
        else:
            previous_position = None
        
        # Clear the canvas if the thumb and small finger  is raised
        if fingers_up == [1, 0, 0, 0, 1]:
            drawing_canvas = np.zeros_like(drawing_canvas)

        # Save the drawing if all fingers are raised except the thumb
        if fingers_up == [0, 1, 1, 1, 1]:
            save_drawing(drawing_canvas)
            analyze_drawing_with_ai(drawing_canvas)
        
    # Overlay the drawing on the frame
    frame = cv2.addWeighted(frame, 1, drawing_canvas, 0.5, 0)

    # Show the processed frames and drawings
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Drawing Canvas", drawing_canvas)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
capture.release()
cv2.destroyAllWindows()
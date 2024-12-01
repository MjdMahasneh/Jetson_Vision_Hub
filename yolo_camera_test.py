from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a pre-trained YOLOv8 nano model

# Open the USB camera (change '0' if using a different device index)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to read frame from the camera.")
        break

    # Perform inference
    results = model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
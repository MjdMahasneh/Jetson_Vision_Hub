import cv2

# Load the Haar cascade file for face detection
cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open the USB camera (adjust 0 if you have multiple cameras)
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

    # Convert frame to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Detector", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
4. Run the Script
Run the script from your terminal:
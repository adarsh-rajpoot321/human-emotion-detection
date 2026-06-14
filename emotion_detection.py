import cv2
from fer import FER

# Start webcam
cap = cv2.VideoCapture(0)

# Load emotion detector
detector = FER(mtcnn=True)

while True:
    
    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect emotions
    emotions = detector.detect_emotions(frame)

    for face in emotions:

        x, y, w, h = face["box"]

        # Get dominant emotion
        emotion, score = max(
            face["emotions"].items(),
            key=lambda item: item[1]
        )

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        # Display emotion
        text = f"{emotion} {score:.2f}"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "Human Emotion Detection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
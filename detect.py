import cv2
import pyttsx3
import time

engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 150)

def speak(msg):
    print("TTS:", msg)
    engine.say(msg)
    engine.runAndWait()

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt.txt",
    "MobileNetSSD_deploy.caffemodel"
)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

cap = cv2.VideoCapture(0)

last_announced_times = {}
cooldown_seconds = 5  # cooldown time for each object

# Tracks when an object was first detected to ensure stability (seconds)
object_first_seen = {}
stable_time = 1.5  # seconds an object must persist before announcement

def format_object_list(objects):
    # Format list ["cat", "dog"] as "cat and dog"
    if len(objects) == 1:
        return objects[0]
    else:
        return ", ".join(objects[:-1]) + " and " + objects[-1]

try:
    speak("Camera object narration activated.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        current_time = time.time()
        detected_objects = set()

        # Update detections and track first seen time
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                detected_objects.add(label)
                if label not in object_first_seen:
                    object_first_seen[label] = current_time
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype(int)
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}",
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Remove objects no longer detected from first seen dict
        to_remove = [obj for obj in object_first_seen if obj not in detected_objects]
        for obj in to_remove:
            del object_first_seen[obj]

        # Announce objects which have been stable for stable_time and cooldown passed
        for obj in detected_objects:
            first_seen_time = object_first_seen.get(obj, 0)
            last_announce = last_announced_times.get(obj, 0)
            if current_time - first_seen_time > stable_time and current_time - last_announce > cooldown_seconds:
                speak(f"{obj} detected")
                last_announced_times[obj] = current_time

        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

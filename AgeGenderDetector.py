
import cv2
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Input image path
image_path = os.path.join(current_dir, '*INSERT JPEG PATH HERE*')

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load image.")
else:
    image = cv2.resize(image, (720, 640))
    print("Image Shape:", image.shape)

    # Model file paths
    face1 = os.path.join(current_dir, "opencv_face_detector.pbtxt")
    face2 = os.path.join(current_dir, "opencv_face_detector_uint8.pb")
    age1 = os.path.join(current_dir, "age_deploy.prototxt")
    age2 = os.path.join(current_dir, "age_net.caffemodel")
    gen1 = os.path.join(current_dir, "gender_deploy.prototxt")
    gen2 = os.path.join(current_dir, "gender_net.caffemodel")

    # Load face detection, age, and gender models
    face_net = cv2.dnn.readNet(face2, face1)
    age_net = cv2.dnn.readNet(age2, age1)
    gender_net = cv2.dnn.readNet(gen2, gen1)

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)

    # Perform face detection
    face_net.setInput(blob)
    detections = face_net.forward()

    # Iterate over detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            # Get the coordinates of the bounding box
            x1 = int(detections[0, 0, i, 3] * image.shape[1])
            y1 = int(detections[0, 0, i, 4] * image.shape[0])
            x2 = int(detections[0, 0, i, 5] * image.shape[1])
            y2 = int(detections[0, 0, i, 6] * image.shape[0])

            # Extract the face ROI
            face_roi = image[y1:y2, x1:x2]

            # Preprocess the face ROI for age and gender estimation
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Perform age and gender estimation
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age_idx = age_preds[0].argmax()
            age_label = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"][age_idx]

            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = "Male" if gender_preds[0][0] > 0.5 else "Female"

            # Draw the bounding box and annotations on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Gender: {gender}, Age: {age_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the annotated image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
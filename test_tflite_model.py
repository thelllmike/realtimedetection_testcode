import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image, input_shape):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def process_output(output_data, threshold=0.5):
    boxes, classes, scores = [], [], []
    for detection in output_data[0]:
        score = detection[4]
        if score > threshold:
            boxes.append(detection[:4])
            classes.append(detection[5])
            scores.append(score)
    return boxes, classes, scores

# Load the TFLite model and allocate tensors
model_path = "model.tflite"  # Ensure the path is correct relative to your script
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Set up Matplotlib figure
plt.ion()
fig, ax = plt.subplots()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    input_data = preprocess_image(frame, input_shape)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output
    boxes, classes, scores = process_output(output_data)

    # Draw the results on the frame
    for i in range(len(boxes)):
        box = boxes[i]
        class_id = int(classes[i])
        score = scores[i]

        ymin, xmin, ymax, xmax = box
        start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
        end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
        color = (0, 255, 0)  # Green
        thickness = 2

        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        label = f'Class: {class_id}, Score: {score:.2f}'
        frame = cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Display the resulting frame
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
plt.close()

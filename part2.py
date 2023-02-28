
''''
This code uses the onnxruntime library to load the ONNX model and perform real-time inference on video frames captured from a webcam.
 The code resizes each frame to 256x256, runs the model on the frame, and post-processes the output to generate a pothole detection map. 
 The detection results are drawn on the frame and displayed in a window using OpenCV. 
 The code continues running until the user presses the "q" key, at which point the video capture is released and the window is closed.
Note that this code is just a sample to give you an idea of how you might use Intel One API for real-time pothole detection. 
The specific details of your application, such as the model architecture, input and output shapes, and post-processing code, will depend on your specific requirements and the data you are using. You may need to modify this code to fit your specific use case.


'''

import cv2
import numpy as np
import onnxruntime
import time

# Load the ONNX model
ort_session = onnxruntime.InferenceSession("pothole_detection_model.onnx")
input_name = ort_session.get_inputs()[0].name

# Open the video capture
cap = cv2.VideoCapture(0)

# Loop over each frame of the video
while True:
    # Capture the next frame
    ret, frame = cap.read()

    # Pre-process the frame
    input_data = cv2.resize(frame, (256, 256))
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    # Run the model on the frame
    start_time = time.time()
    ort_session.run(input_name, input_data)
    inference_time = time.time() - start_time

    # Post-process the output
    pothole_detection_map = ort_session.get_outputs()[0]

    # Draw the detection results on the frame
    for i in range(pothole_detection_map.shape[0]):
        for j in range(pothole_detection_map.shape[1]):
            if pothole_detection_map[i, j] >= 0.5:
                cv2.rectangle(frame, (j * 256, i * 256), ((j + 1) * 256, (i + 1) * 256), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

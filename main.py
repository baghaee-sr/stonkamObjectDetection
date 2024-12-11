import tensorflow as tf
import numpy as np
import cv2
import time

def load_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(output['index']) for output in output_details]
    return outputs

def preprocess_frame(frame, input_shape, input_dtype):
    original_height, original_width = frame.shape[:2]
    target_height, target_width = 480, 640
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized_model_frame = cv2.resize(padded_frame, (input_shape[1], input_shape[2]))
    if input_dtype == np.uint8:
        input_data = np.expand_dims(resized_model_frame, axis=0).astype(np.uint8)
    else:
        input_data = np.expand_dims(resized_model_frame / 255.0, axis=0).astype(np.float32)
    return input_data, padded_frame

def draw_detections(image, boxes, classes, scores, threshold=0.5):
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            start_point = (int(xmin * width), int(ymin * height))
            end_point = (int(xmax * width), int(ymax * height))
            label = f"Class {int(classes[i])}: {scores[i]:.2f}"
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 10)
            cv2.putText(image, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    tflite_model_path = "models/efficientdet-lite0.tflite"
    camera_ip = "192.168.88.10"
    stream_url = f"rtsp://{camera_ip}/mainstream"
    interpreter = load_model(tflite_model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video stream from IP camera.")
        return
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video stream.")
            break
        input_data, processed_frame = preprocess_frame(frame, input_shape, input_dtype)
        outputs = run_inference(interpreter, input_data)
        boxes = outputs[0][0]
        classes = outputs[1][0]
        scores = outputs[2][0]
        output_frame = draw_detections(processed_frame, boxes, classes, scores, threshold=0.5)
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Live Object Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

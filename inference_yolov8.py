from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

image_path = "test1.webp"
# Run batched inference on a list of images
results = model(image_path,
                conf = 0.3)  # return a list of Results objects
print("results_list_len: =", len(results))
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print("boxes: ",boxes)
    # masks = result.masks  # Masks object for segmentation masks outputs
    # print("masks: ", masks.shape)
    keypoints = result.keypoints  # Keypoints object for pose outputs
    print("keypoints: ", keypoints)
    probs = result.probs  # Probs object for classification outputs
    print("prob: ",probs)
    # obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk

# Run inference on 'bus.jpg' with arguments
model.predict(image_path,
               save=True, imgsz=640, conf=0.3)
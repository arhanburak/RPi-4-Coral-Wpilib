import cv2
import numpy as np
import time
import ntcore

from cscore import CameraServer
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


def main():
    model = "model.tflite"
    labels_file_path = "labels.txt"

    print("Starting Camera Server...")
    cs = CameraServer
    cs.enableLogging()

    # Capture from the first USB Camera on the system
    camera = cs.startAutomaticCapture()
    camera.setResolution(1920, 1080)

    # Get a CvSink. This will capture images from the camera
    cvSink = cs.getVideo()

    # Setup a CvSource. This will send images back to the Dashboard
    outputStream = cs.putVideo("Processed", 1920, 1080)

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)

    # Load the model and labels here
    print('Loading {} with {} labels.'.format(model, labels_file_path))
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)
    labels = read_label_file(labels_file_path)

    # Wait for NetworkTables to start
    time.sleep(0.5)

    prev_time = time.time()
    print("Vision Processing started...")

    while True:
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error, notify the output.
        frame_time, img = cvSink.grabFrame(img)
        if frame_time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # skip the rest of the current iteration
            continue

        # OpenCV image processing logic.
        cv2_im = img
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, 0.66)
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)


        # Calculate the processing performance by gathering Frames Per Second data.
        processing_time = time.time() - prev_time
        prev_time = time.time()
        fps = 1 / processing_time
        cv2.putText(cv2_im, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Send some image back to the dashboard
        outputStream.putFrame(img)

# Draw a bounding box around the detected objects and label them.         
def append_objs_to_img(cv2_im, inference_size, objs, labels, dblTopic: ntcore.DoubleTopic):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:

        # Create a NetworkTableEntry for each label
        nt = NetworkTableInstance.getDefault()
        cone_coordinates_table = nt.getTable("cone_coordinates")
        cone_coordinates_entry = cone_coordinates_table.getDoubleTopic("cone_coordinates")
        cube_coordinates_table = nt.getTable("cube_coordinates")
        cube_coordinates_entry = cube_coordinates_table.getDoubleTopic("cube_coordinates")
        merge_coordinates_table = nt.getTable("merge_coordinates")
        merge_coordinates_entry = merge_coordinates_table.getDoubleTopic("merge_coordinates")

        try:
            dblPub = dblTopic.publish()
            dblPub([center_x, center_y])
        except Exception as e:
            print(e)
            pass

        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        # Calculate the center coordinates of each detected object
        center_x = (obj.bbox.xmin + obj.bbox.xmax) / 2
        center_y = (obj.bbox.ymin + obj.bbox.ymax) / 2

        # Publish the center coordinates to the appropriate NetworkTableEntry based on the label
        if obj.id == "cone":
            cone_coordinates_entry.dblPub([center_x, center_y])
        elif obj.id == "cube":
            cube_coordinates_entry.dblPub([center_x, center_y])
        elif obj.id == "merge":
            merge_coordinates_entry.dblPub([center_x, center_y])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


if __name__ == '__main__':
    main()
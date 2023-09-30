from cscore import CameraServer
from ntcore import NetworkTableInstance, EventFlags

import cv2
import json
import numpy as np
import time

# Python code to run the classifier:
import re
import os
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

# the TFLite converted to be used with edgetpu
modelPath = "model.tflite"
# The path to labels.txt that was downloaded with your model
labelPath = "labels.txt"

team = None
server = False

# This function takes in a TFLite Interpter and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return classify.get_classes(interpreter)


def main():
    with open('/boot/frc.json') as f:
        config = json.load(f)
    camera = config['cameras'][0]

    width = camera['width']
    height = camera['height']

    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    # Table for vision output information
    # start NetworkTables
    ntinst = NetworkTableInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient4("wpilibpi")
        #ntinst.setServerTeam(team)
        ntinst.startDSClient()
    #vision_nt = NetworkTables.getTable('Vision')
    vision_nt = ntinst.getTable('Vision')

    # Load your model onto the TF Lite Interpreter
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)

    # Wait for NetworkTables to start
    time.sleep(0.5)

    prev_time = time.time()
    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)
        output_img = np.copy(input_img)

        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        # Flip image so it matches the training input
        ##frame = cv2.flip(frame, 1)
        # Classify and display image
        results = classifyImage(interpreter, input_img)
        confidence = round(100*results[0].score, 3)
        tagline = f"{labels[results[0].id]} ({confidence} %)"
        print(f'Label: {labels[results[0].id]}, Score: {results[0].score}')
        cv2.putText(output_img, tagline, (0, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # TODO: Output results to the Network Table

        processing_time = start_time - prev_time
        prev_time = start_time

        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        output_stream.putFrame(output_img)

main()
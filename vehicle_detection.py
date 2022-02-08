import cv2
import numpy as np
import utility as utils


class VehicleDetection:
    def __init__(self, vehicle_net_weights, vehicle_net_cfg, vehicle_detection_threshold=.5):
        self.vehicle_detection_model_weights = vehicle_net_weights
        self.vehicle_detection_model_cfg = vehicle_net_cfg
        self.vehicle_detection_threshold = vehicle_detection_threshold
        self.vehicle_detection_model = None

    def load_vehicle_detection_model(self):
        self.vehicle_detection_model = cv2.dnn.readNetFromDarknet(self.vehicle_detection_model_cfg,
                                                                  self.vehicle_detection_model_weights)
        self.vehicle_detection_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.vehicle_detection_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def run_vehicle_detection(self, image):
        # Create a 4D blob from a image.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.vehicle_detection_model.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.vehicle_detection_model.forward(utils.getOutputsNames(self.vehicle_detection_model))

        # Remove the bounding boxes with low confidence
        R = utils.postprocess(image, outs, self.vehicle_detection_threshold)

        Icars = []
        Lcars = []

        if len(R):
            WH = np.array(image.shape[1::-1], dtype=float)
            for i, r in enumerate(R):
                # if classId in ['car', 'bus'] and confidence > vehicle_threshold
                if r[0] in [6, 7] and r[1] > self.vehicle_detection_threshold:
                    box = r[2]
                    x1, y1, x2, y2 = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
                    tl = np.array([x1, y1])
                    br = np.array([x2, y2])
                    label = utils.Label(0, tl, br)
                    Lcars.append(label)
                    Icar = utils.crop_region(image, label)
                    Icars.append(Icar.astype(np.uint8))

        return Icars, Lcars

from tensorflow.keras.models import model_from_json
import utility as utils
import numpy as np


class NumberPlateDetection:
    def __init__(self, number_plate_detection_net_weights, number_plate_detection_net_json,
                 number_plate_detection_threshold = .6):
        self.number_plate_net_weights = number_plate_detection_net_weights
        self.number_plate_net_json = number_plate_detection_net_json
        self.number_plate_detection_threshold = number_plate_detection_threshold
        self.number_plate_detection_model = None

    def load_number_plate_detection_model(self):
        # Load the model
        with open(self.number_plate_net_json, 'r') as json_file:
            wpod_json = json_file.read()
        self.number_plate_detection_model = model_from_json(wpod_json)
        self.number_plate_detection_model.load_weights(self.number_plate_net_weights)

    def run_number_plate_detection(self, vehicle_img):
        ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)
        Llps, LlpImgs, elapsed = utils.detect_lp(self.number_plate_detection_model, utils.im2single(vehicle_img),
                                                 bound_dim, 2 ** 4, (240, 80),
                                                 self.number_plate_detection_threshold)

        Ilps = []
        for LlpImg in LlpImgs:
            Ilp = LlpImg * 255.
            Ilps.append(Ilp.astype(np.uint8))

        return Llps, Ilps, elapsed
from vehicle_detection import VehicleDetection
from number_plate_detection import NumberPlateDetection
import utility as utils
from config_data import ConfigData
import cv2


class NumberPlateRecognition:
    def __init__(self):
        self.ocr_weights = ConfigData.ocr_weights_path
        self.ocr_cfg = ConfigData.ocr_config_path
        self.ocr_classes_path = ConfigData.ocr_classes_path
        self.ocr_classes = None
        self.ocr_threshold = ConfigData.ocr_threshold
        self.ocr_model = None
        self.number_plates = []
        self.vehicle_detection = VehicleDetection(ConfigData.vehicle_net_weights, ConfigData.vehicle_net_cfg)
        self.vehicle_detection.load_vehicle_detection_model()
        self.number_plate_detection = NumberPlateDetection(ConfigData.wpod_lp_weights_path,
                                                           ConfigData.wpod_lp_json_path)
        self.number_plate_detection.load_number_plate_detection_model()

    def load_ocr_model(self):
        self.ocr_model = cv2.dnn.readNetFromDarknet(self.ocr_cfg, self.ocr_weights)
        self.ocr_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.ocr_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.ocr_classes = self.get_ocr_classes()

    def get_ocr_classes(self):
        with open(self.ocr_classes_path, 'rt') as f:
            ocr_classes = f.read().rstrip('\n').split('\n')

        return ocr_classes

    def run_number_plate_recognition(self, input_image):
        if isinstance(input_image, str):
            input_image = cv2.imread(input_image)
        vehicle_images, _ = self.vehicle_detection.run_vehicle_detection(input_image)
        for vehicle_image in vehicle_images:
            _, number_plate_images, _ = self.number_plate_detection.run_number_plate_detection(vehicle_image)
            for number_plate_img in number_plate_images:
                h, w, _ = number_plate_img.shape
                # Create a 4D blob from a frame
                blob = cv2.dnn.blobFromImage(number_plate_img, 1 / 255, (240, 80), [0, 0, 0], 1, crop=False)
                # Sets the input to the network
                self.ocr_model.setInput(blob)
                # Runs the forward pass to get output of the output layers
                outs = self.ocr_model.forward(utils.getOutputsNames(self.ocr_model))
                # Remove the bounding boxes with low confidence
                R = utils.postprocess(number_plate_img, outs, self.ocr_threshold, None)

                lp_str = ''
                if len(R):
                    L = utils.dknet_label_conversion(R, w, h, self.ocr_classes)
                    L.sort(key=lambda x: x.tl()[0])
                    lp_str = ''.join([chr(l.cl()) for l in L])
                    self.number_plates.append(lp_str)
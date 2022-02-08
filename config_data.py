from dataclasses import dataclass
import os


@dataclass
class ConfigData:
    # Initialize vehicle detection parameters
    vehicle_threshold: float = .5
    vehicle_net_weights: str = os.path.join(os.path.dirname(__file__),
                                            'models/vehicle-detector/yolo-voc.weights')
    vehicle_net_cfg: str = os.path.join(os.path.dirname(__file__), 'models/vehicle-detector/yolo-voc.cfg')

    # Initialize number plate detection parameters
    lp_threshold: float = .6
    wpod_lp_weights_path: str = os.path.join(os.path.dirname(__file__), 'models/lp-detector/wpod-net_update1.h5')
    wpod_lp_json_path: str = os.path.join(os.path.dirname(__file__), 'models/lp-detector/wpod-net_update1.json')

    # Initialize ocr parameters
    ocr_threshold: float = .4
    ocr_weights_path: str = os.path.join(os.path.dirname(__file__), 'models/ocr/ocr-net.weights')
    ocr_config_path: str = os.path.join(os.path.dirname(__file__), 'models/ocr/ocr-net.cfg')
    ocr_classes_path: str = os.path.join(os.path.dirname(__file__), 'models/ocr/ocr-net.names')

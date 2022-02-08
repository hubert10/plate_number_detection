# plate_number_detection

Detects license plate of car and recognizes its characters

## Dependencies

1. python 3.7
2. python opencv and opencv-contrib-python 4.2.0
3. Tensorflow 2.7
4. numpy

## Create your Virtualenv and install the requirements
python -m venv myenv

## Activate the created Virtualenv
source myenv/bin/activate

## Install the requirements

pip install -r requirements.txt

## Test

1. Install al the above dependencies
1. Download or git clone the repository ap_number_plate_recognition_identification to your system and copy the package number_plates to the desired folder
2. To get the licence plates from an image, you have to provide the path to image while running the following lines of code:
```
# import the module
from number_plates.number_plate_recognition import NumberPlateRecognition
# instantiate the number plate recognizer object
np_recognizer = NumberPlateRecognition()
# load the recognition model
np_recognizer.load_ocr_model()
# run the recognition on your image
np_recognizer.run_number_plate_recognition(path_to_your_image)
# obtain a list of licence plates present on the image
licence_plates = np_recognizer.number_plates
```
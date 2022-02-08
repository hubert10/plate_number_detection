# import the module
from number_plate_recognition import NumberPlateRecognition

# instantiate the number plate recognizer object
np_recognizer = NumberPlateRecognition()
# load the recognition model
np_recognizer.load_ocr_model()
# and apply it to your image
np_recognizer.run_number_plate_recognition(
    "/home/hubert/Desktop/AI/number_plates/images/test_img1.jpeg"
)
# obtain a list of licence plates present on the image
licence_plates = np_recognizer.number_plates
print(licence_plates)

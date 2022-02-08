# plate_number_detection

Detects license plate of car and recognizes its characters

## Dependencies

1. python 3.7
2. opencv-contrib-python 4.2.0
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
1. Download or git clone the repository plate_number_detection to your system and copy the package number_plates to the desired folder
3. Download the pre-trained models fron google-drive:
[Download Pre-trained models](https://drive.google.com/drive/folders/1xKrakmTFMN8KbLU26h-CTH0InxnXvl0F?usp=sharing)
4. Unzip the models.zip folder and place it into the root folder of your project

5. To get the licence plates from an image, you have to provide the path to image while running the following lines of code:
```
# python plate.py
```
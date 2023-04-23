# Vegnet Application


## Problem Statement

Although there are pre-trained models available for detecting vegetables on the ImageNet dataset, it remains unclear whether the accuracy of these models holds true for other datasets. Additionally, these models lack the ability to identify vegetables like tomato and predict important attributes such as the age, ripeness, damage, unripeness, or dryness of the vegetable.

## Dataset

## Vegetables:
Tomato<br>
Bell Peper<br>
New Mexico Green Chile<br>
Chile Peper<br>

## Vegetables are classified as:
Damaged<br>
Dried<br>
Old<br>
Ripe<br>
Unripe<br>
<br><br>
Accuracy of pre-existing models is calculated on VegNet dataset. Then training and testing is done to detect the vegetable along with its quality.

## Packages to be installed:
python3-pip<br>
python3-setuptools<br>
python3-tk<br>
python3-wheel<br>
python3.9-distutils<br>
latest pip3<br>
keras-applications 1.0.8<br>
numpy 1.23.5<br>
pillow 9.5.0<br>
pip 23.1<br>
tensorflow 2.12.0<br>
tensorflow-datasets 4.9.2<br>

## File structure:
### Dataset is categorised in folders as follows:
![image](https://user-images.githubusercontent.com/82139597/233821904-fa71e7e3-2247-45fa-a3e4-ce2ff974dc2c.png)

### VC_Classname structure:
![image](https://user-images.githubusercontent.com/82139597/233821995-674b9a7e-116e-49d0-9b49-95843bfcff27.png)

### VQC_Classname structure:
![image](https://user-images.githubusercontent.com/82139597/233821964-51eb663a-3f98-4ef5-8982-d855834e3a9f.png)

## Commands to run code:
Use the following commands to run code from the Code Ocean:
python3 -u 'Code/VC_Xception Model.py' "$@"
python3 -u 'Code/VQC_Xception Model.py' "$@"





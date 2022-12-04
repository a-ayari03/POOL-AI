# POOL-AI

The aim of the project is to make it possible to detect swimming pools using satellite images. This detection is done in two steps, on the one hand, a call to the google maps api allows to obtain an aerial photograph on the GPS coordinates or the postal address concerned. In a second step, a customised model using the Yolov5 architecture is used to verify the presence of one or more swimming pools on the photograph.

### Installation

The installation of python3 and pip is necessary for the installation of the packages.

    pip install -r requirements.txt

### 1. Dataset

The constitution of the dataset allowing the training of the yolov5 model was made possible with the public data of the land register available on the following link: https://cadastre.data.gouv.fr/datasets/cadastre-etalab. Thus, it was possible to filter the GPS coordinates of the declared residential plots with a swimming pool (only on the commune of VAR). The dataset is composed of 800 aerial photos coming from a call to the google maps API and labelled with the data annotation tool Labelstudio.
The notebook [Get_Dataset.ipynb](https://github.com/a-ayari03/POOL-AI/blob/main/build_pool_model/Get_Dataset.ipynb "Get_Dataset.ipynb")/** allows to reconstitute the initial data set.

### 2. Model

YOLOv5 is one of the recent versions of the YOLO model family. YOLOv5 is the first YOLO model to have been written in Pytorch. It also allows to answer object detection and classification problems. The transfer learning method was used to capitalize on all the power of the Yolo model. Thus, the last layers of neurons have been frozen to get the maximum performance and to exploit the knowledge acquired before.
The notebook [Pool_object_detection.ipynb](https://github.com/a-ayari03/POOL-AI/blob/main/build_pool_model/Pool_object_detection.ipynb "Pool_object_detection.ipynb") allows to reproduce the model generated for pool detection. It requires however the use of a GPU and the labelled dataset (available).

### 3. Detection
Detection is performed on the notebook [detection.ipynb](https://github.com/a-ayari03/POOL-AI/blob/main/detection.ipynb "detection.ipynb"). It requires a GPU and the results will be saved in the 'inference' folder.

### Authors and acknowledgment
Alexandre Ayari, alexandre.ayari03@gmail.com,
Théo Bardon, theo.bardon@gmail.com




# Wise Transportation

It is the official repo for "Multiple-Model Ensemble Learning for Wrong-Way-Cycling Prediction
in Long-Form Video"

## Requirements
First, change the working directory to ``mmyolo/``
```
cd mmyolo
```
Install MMEngine, MMCV and MMDetection using MIM
```shell
pip install -U openmim
mim install -r requirements/mminstall.txt
```
Install MMYOLO
```shell 
mim install -r requirements/albu.txt
mim install "mmyolo"
```

## Data Preparation

Run code in ``dataset_scripts`` to extract frames in videos as you want.  

Run ``mmyolo/detect2label.py`` to generate orientation label by detection model.

The training set are placed at ``data/``, and the folder structure should be:
```
wrong-way-cycling
├── data
│   ├── Dataset_one_class
│   │   ├── Annotations
│   │   │   ├── coco_info.json
│   │   ├── Images
│   │   │   ├── ...
```

## Train & Test
First, change the working directory to ``mmyolo/``
```
cd mmyolo
```
Sample command for training:
```shell
Python tools/train.py configs/custom/5s.py
```

Sample command for testing:
```shell
Python tools/test.py configs/custom/5s.py %path/to/checkpoint.pth% --show-dir %path/to/folder/to/save/results%
```

## Inferencing
Step1. Generate Data from Video
```
python dataset_script\Frame_extraction_2_monte.py --name %NAME% --Eg %Eg%
```

Step2. Run Inferencing Script
```bash
cd mmyolo
# Ensemble Method
python detect_anglepred_pipeline.py --name %NAME% --Eg %Eg%
# Orientation-aware Model Method
python anglepred_pipeline.py --name %NAME% --Eg %Eg%
# Detection-based Method 
python detect_pipeline.py --name %NAME% --Eg %Eg%
```

## License

Please check the LICENSE file.
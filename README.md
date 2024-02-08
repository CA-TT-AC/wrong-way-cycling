# Wrong-way-cycling prediction

It is the official repo for "Multiple-Model Ensemble Learning for Wrong-Way-Cycling Prediction
in Long-Form Video". Welcome to report issue or email author to report your problem.

## Requirements
First, change the working directory to ``mmyolo/``
```
cd mmyolo
```
Install Python, Pytorch (Tested On Python=3.10, Pytorch=2.0.0, cuda=11.7)
```
conda create -n wwc python=3.10
conda activate wwc
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
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
All the datasets are available in https://huggingface.co/datasets/CATTAC/wrong-way-cycling!

<!-- Run code in ``dataset_scripts`` to extract frames in videos as you want.  

Run ``mmyolo/detect2label.py`` to generate orientation label by detection model.

The training set are placed at ``data/``, and the folder structure should be:
```
wrong-way-cycling
├── data
│   ├── Dataset
│   │   ├── Annotations
│   │   │   ├── coco_info.json
│   │   ├── Images
│   │   │   ├── ...
``` -->
## Orientation-aware Model Training
First, change the working directory to ``angle_prediction/``
```
cd angle_prediction
```
Both pretraining and finetuning are done by ``main.py``.  
For pretraining, set ``--data_path`` to pretraining dataset path.  
For finetuning, set ``--data_path`` to finetuning dataset path, and set ``--resume`` to the path of your initialized checkpoint.  
Sample command for training:
```shell
python main.py --data_path %path/to/dataset/ --resume %path/to/checkpoint/
```
## Detection Model Training & Testing
First, change the working directory to ``mmyolo/``
```
cd mmyolo
```
Sample command for training:
```shell
python tools/train.py configs/custom/5s.py
```

Sample command for testing:
```shell
python tools/test.py configs/custom/5s.py %path/to/checkpoint.pth% --show-dir %path/to/folder/to/save/results%
```
Modified the configuration in `%you_model%.py`, configuration like train and test datasets path or some hyperparameters are in the python file.

## Whole Pipeline Inferencing
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
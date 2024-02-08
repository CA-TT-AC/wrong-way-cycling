# Model
The model checkpoints have been hosted on the Hugging Face platform.
[Hugging Face](https://huggingface.co/ascka/MMYolo/tree/main)
- Install Python, Pytorch (Tested On Python=3.10, Pytorch=2.0.0, cuda=11.7)
```
conda create -n wwc python=3.10
conda activate wwc
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```
- Install MMEngine, MMCV, and MMDetection using MIM
```shell
pip install -U openmim
mim install -r requirements/mminstall.txt
```
- Install MMYOLO
```shell 
mim install -r requirements/albu.txt
mim install "mmyolo"
```

# Train & Test
- Modified the configuration in `%you_model%.py`, configuration like train and test datasets path or some hyperparameters are in the python file.
- Training
Place the training dataset in the 'data' folder with the following structure:
   - Dataset_one_class
     - Annotations
       - coco_info.json
     - Images
       - ...
```shell
python tools/train.py configs/custom/5s.py
```
If training is interrupted, you can resume it by adding ```--resume``` to the training command. The program will automatically resume training from the latest saved checkpoint in the 'work_dirs' directory.

- Testing
```shell
python tools/test.py configs/custom/5s.py %path/to/checkpoint.pth% --show-dir %path/to/folder/to/save/results%
```
Running the above test command will provide you with AP performance metrics, as well as automatically save inference result images to the 'work_dirs/king/{timestamp}/show_results' folder.

# Inference
- Object Detection from Images
```python
    from infer_image import image_infer
    data = image_infer(img, path_to_config, path_to_checkpoint, threshold)
    bbox = data.bboxes
    """
    print(data)
    <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7fb492de6280>
    print(bbox)
    tensor([[0.4997, 0.7707, 0.0595, 0.4188],
            [0.8101, 0.3105, 0.5123, 0.6263]])
    """

```

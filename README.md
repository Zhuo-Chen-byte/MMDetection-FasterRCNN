# MMDetection-FasterRCNN #
## Content ##
In addition to all files of a regular mmdetection project, this directory holds (*after you download them*):  
- Two pretrained PyTorch faster rcnn backbones (*faster_rcnn_r50_fpn_1x_coco.pth* and *faster_rcnn_r50_fpn_2x_coco.pth* under pretrained/)  
- The complete configuration to train a faster-rcnn model (*my_faster_rcnn.py* under configs/faster_rcnn/)  

To learn other contents this directory holds, see *https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md*

## Environment ##
Below is the complete environment setup to run a faster-rcnn:  
- Linux
- Python 3.7.11
- Pytorch 1.5.0
- TorchVision: 0.6.0
- OpenCV: 4.5.4
- GCC 9.2.0
- CUDA 10.2
- CuDNN 7.6.5
- MMCV: 1.2.7
- MMCV Compiler: GCC 7.3
- MMCV CUDA Compiler: 10.2
- MMDetection: 2.10.0

## Installation ##
### Prepare the environment ###
1. Create a conda virtual environment and activate it

    ```shell
    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    ```
    
2. Install PyTorch and torchvision

    ```shell
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
    ```
    
### Install the directory ###
1. Install mmcv

    The template command to install mmcv is
    
    ```shell
    pip install mmcv -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```
    
    For the environment this directory depends on specifically, you can run
    ```shell
    pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html
    ```
    
2. Install this mmdetection  

    You can clone the directory by running this command
  
    ```shell
    git clone https://github.com/Zhuo-Chen-byte/MMDetection-FasterRCNN.git
    ```
    
    or down the .zip manually
    
3. Prepare mmdet

    After you download this directory, simply run the following command:
    
    ```shell
    cd MMDetection-FasterRCNN
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```
4. Verification
    
    You can run the following commands to check if installation is successfully
    
    ```shell
    from mmdet.apis import init_detector, inference_detector
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'pretrained/faster_rcnn_r50_fpn_1x_coco.pth'
    device = 'cuda:0'
    
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    inference_detector(model, 'demo/demo.jpg')
    ```
    
    The above commands shall be compiled without error if installation is successful


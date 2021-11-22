# MMDetection-FasterRCNN #
## Content ##
In addition to all files of a regular mmdetection project, this directory holds (*after you download them*):  
- Two pretrained PyTorch faster rcnn backbones (*faster_rcnn_r50_fpn_1x_coco.pth* and *faster_rcnn_r50_fpn_2x_coco.pth* under *pretrained/*)  
- The complete configuration to train a faster-rcnn model (*my_faster_rcnn.py* under *configs/faster_rcnn/*)  

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

## Training Your Own Faster-RCNN ##
1. Prepare the dataset
    To run a faster-rcnn based on this mmdetection, you need a coco dataset
    To learn more about a coco dataset, see *https://github.com/cocodataset*
   
2. Edit the configuration  
    After preparing the dataset, look at *configs/my_faster_rcnn.py*  
    Take a look at these lines:  
    
    At *line 46*,
    
    ```shell
    num_classes=1
    ```
    
    records the number of classes you want the model to classify. Before training, changes *1* to the number of classes you need  
    
    At *line 153 & 154, 177 & 178, and 201 & 202*, 
    
    ```shell
    ann_file='/data/zhuochen/coco/annotations/train_val.json'
    img_prefix='/data/zhuochen/coco/train_images/' # line 154
    
    ann_file='/data/zhuochen/coco/annotations/test_val.json'
    img_prefix='/data/zhuochen/coco/test_images/' # line 178
    
    ann_file='/data/zhuochen/coco/annotations/test_val.json'
    img_prefix='/data/zhuochen/coco/test_images/' # line 202
    ```
    
    are the addresses of the train, test, and validation jsons and images. Change them to your own before training  
    
    Or you can change *line 107*
    
    ```shell
    data_root = '/data/zhuochen/coco/'
    ```
    which is the root directory of your coco dataset and uses
    
    ```shell
    ann_file= data_root + annotation_directory
    img_prefix = data_root + image_directory
    ```
    
    to access your dataset
    
    Finally, from *line 232* to *line 243*
    
    ```shell
    runner = dict(type='EpochBasedRunner', max_epochs=15)
    checkpoint_config = dict(create_symlink=False, interval=1)
    log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
    custom_hooks = [dict(type='NumClassCheckHook')]
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    load_from = 'pretrained/faster_rcnn_r50_fpn_2x_coco.pth'
    resume_from = None
    workflow = [('train', 1), ('val', 1)]
    classes = ('cancer', )
    work_dir = '/mnts2d/med_data1/zhuochen'
    gpu_ids = range(0, 1)
    ```
    
    are other training parameters:
    1. *max_epochs* at *line 232* determines the total number of epoches to train
    2. *load_from* at *line 238* is the directory which stores the pretrained model parameters 
    3. *classes* at *line 241* are the complete classification categories based on your dataset
    4. *work_dir* at *line 242* is the directory to store your training results  
 
    Changes these parameters based on your dataset

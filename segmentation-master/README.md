# High Remote Sence Image Semantic Segmentation in PyTorch

<!-- TOC -->

- [Semantic Segmentation in PyTorch](#semantic-segmentation-in-pytorch)
  - [Requirements](#requirements)
  - [Main Features](#main-features)
    - [Models](#models)
    - [Datasets](#datasets)
    - [Losses](#losses)
    - [Learning rate schedulers](#learning-rate-schedulers)
    - [Data augmentation](#data-augmentation)
  - [Training](#training)
  - [Inference_end_to_end](#inference_end_to_end)
  - [Code structure](#code-structure)
  - [Config file format](#config-file-format)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

This repo contains a PyTorch an implementation of different semantic segmentation models for different datasets.

## Requirements Enviorment
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.

```bash
conda env create -f py3.5.yaml
```

```bash
pip install -r requirements.txt
```

## Main Features

- A clear and easy to navigate structure,
- A `json` config file with a lot of possibilities for parameter tuning,
- Supports various models, losses, Lr schedulers, data augmentations and datasets,

**So, what's available ?**

### Models 
- (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611)
- (**GCN**) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [[Paper]](https://arxiv.org/abs/1703.02719)
- (**UperNet**) Unified Perceptual Parsing for Scene Understanding [[Paper]](https://arxiv.org/abs/1807.10221)
- (**DUC, HDC**) Understanding Convolution for Semantic Segmentation [[Paper]](https://arxiv.org/abs/1702.08502) 
- (**PSPNet**) Pyramid Scene Parsing Network [[Paper]](http://jiaya.me/papers/PSPNet_cvpr17.pdf) 
- (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[Paper]](https://arxiv.org/abs/1606.02147)
- (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper]](https://arxiv.org/abs/1505.04597)
- (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [[Paper]](https://arxiv.org/pdf/1511.00561)
- (**FCN**) Fully Convolutional Networks for Semantic Segmentation (2015): [[Paper]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 

### Datasets

- **xiongan** data in xiongan

### Losses
In addition to the Cross-Entorpy loss, there is also
- **Dice-Loss**, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize.
- **CE Dice loss**, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.
- **Focal Loss**, an alternative version of the CE, used to avoid class imbalance where the confident predictions are scaled down.
- **Lovasz Softmax** lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses (for more details, check the paper: [The Lovász-Softmax loss](https://arxiv.org/abs/1705.08790)).

### Learning rate schedulers
- **Poly learning rate**, where the learning rate is scaled down linearly from the starting value down to zero during training. Considered as the go to scheduler for semantic segmentaion (see Figure below).
- **One Cycle learning rate**, for a learning rate LR, we start from LR / 10 up to LR for 30% of the training time, and we scale down to LR / 25 for remaining time, the scaling is done in a cos annealing fashion (see Figure bellow), the momentum is also modified but in the opposite manner starting from 0.95 down to 0.85 and up to 0.95, for more detail see the paper: [Super-Convergence](https://arxiv.org/abs/1708.07120). 


### Data augmentation
All of the data augmentations are implemented using OpenCV in `\base\base_dataset.py`, which are: rotation (between -10 and 10 degrees), random croping between 0.5 and 2 of the selected `crop_size`, random h-flip and blurring

## Training
To train a model, first get the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (in the config.json file is detailed below), then simply run:

```bash
python train.py --config config.json
```
In training ,if use deeplabv3plus,maybe epoch=40 is enough,but for PSPNet,epoch=50 of more.

The training will automatically be run on the GPUs (if more that one is detected and  multipple GPUs were selected in the config file, `torch.nn.DataParalled` is used for multi-gpu training), if not the CPU is used. The log files will be saved in `saved\runs` and the `.pth` chekpoints in `saved\`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```

## Inference_end_to_end

For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference_end_to_end.py --config config.json --model best_model.pth --images images_folder
```
```bash
exp:python inference_end_to-end.py -i './data/images' -m './best_model_pspnet_3.pth' 
```

The predictions will be saved as `.png` images using the default palette in the passed fodler name, if not, `outputs\` is used

Here are the parameters availble for inference(and some in config.json "infernece"):
```
--config       The config file used for training the model.
```

**Trained Model:**

| Model     | Backbone     | xiongan train acc  | xiongan val acc| xiongan test acc|
| :-------- | :----------: |:-----------------: |:---------------:|:------------:|
| PSPNet    | ResNet 152    | 95%                | 93%            | 92%         |



## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── inference.py - inference using a trained model
  ├── trainer.py - the main trained
  ├── config.json - holds configuration for training
  │
  ├── data/
  │   ├── images/-tiff images path
  │   ├── labels/ - tiff images label RGB image
  │   └── images_jpg/-jpg image path
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │ 
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```

## Config file format
Config files are in `.json` format:
```javascript
{
  "name": "PSPNet",         // training session name
  "n_gpu": 1,               // number of GPUs to use for training.
  "use_synch_bn": true,     // Using Synchronized batchnorm (for multi-GPU usage)
  "num_classes": 8,         // num_classes
  "palette": [0, 0, 0, 150, 250, 0, 0, 250, 0, 0, 100, 0, 200,
               0, 0, 255, 255, 255, 0, 0, 200, 0, 150, 250],  //palette to trans RGB to 1 channel 0-->8
    "arch": {
        "type": "PSPNet", // name of model architecture to train(PSPNet,DeepLab,SegNet,GCN and so on)
        "args": {
            "in_channels": 3,   //in channels
            "backbone": "resnet50",     // encoder type type
            "freeze_bn": false,         // When fine tuning the model this can be used
            "freeze_backbone": false    // In this case only the decoder is trained
        }
    },

    "train_loader": {
        "type": "xiongan",          // Selecting data loader
        "args":{
            "data_dir": "data/",  // dataset path
            "batch_size": 32,     // batch size
            "augment": true,      // Use data augmentation
            "crop_size": 512,     // Size of the random crop after rescaling
            "shuffle": true,
            "base_size": 512,     // The image is resized to base_size, then randomly croped
            "scale": true,        // Random rescaling between 0.5 and 2 before croping
            "flip": true,         // Random H-FLip
            "rotate": true,       // Random rotation between 10 and -10 degrees
            "blur": true,         // Adding a slight amount of blut to the image
            "split": "train_aug", // Split to use, depend of the dataset
            "num_workers": 8
            "need_to_cut":true    //if use tiff to generate dataset (small images)
        }
    },

    "val_loader": {     // Same for val, but no data augmentation, only a center crop
        "type": "xiongan",
        "args":{
            "data_dir": "data/",
            "batch_size": 32,
            "crop_size": 512,
            "val": true,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,      // Using lr/10 for the backbone, and lr for the rest
        "args":{
            "lr": 0.01,               // Learning rate
            "weight_decay": 1e-4,     // Weight decay
            "momentum": 0.9
        }
    },

    "loss": "CrossEntropyLoss2d",     // Loss (see utils/losses.py)
    "ignore_index": 255,              // Class to ignore (must be set to -1 for ADE20K) dataset
    "lr_scheduler": {   
        "type": "Poly",               // Learning rate scheduler (Poly or OneCycle)
        "args": {}
    },

    "trainer": {
        "epochs": 80,                 // Number of training epochs
        "save_dir": "saved/",         // Checkpoints are saved in save_dir/models/
        "save_period": 10,            // Saving chechpoint each 10 epochs
  
        "monitor": "max Mean_IoU",    // Mode and metric for model performance 
        "early_stop": 10,             // Number of epochs to wait before early stoping (0 to disable)
        
        "tensorboard": true,        // Enable tensorboard visualization
        "log_dir": "saved/runs",
        "log_per_iter": 20,         

        "val": true,
        "val_per_epochs": 1         // Run validation each 5 epochs
    }
    "inference": {
        "model_weight_path": "./best_model_pspnet_3.pth",  //weight_path
        "images_dir_path": "./data/images_jpg",            //images to segmentation
        "output_path": "./outputs",                        //segmengt image path to save
        "extension": ".jpg",                               //extension of  images to segment(.jpg,.tiff,.tif)
        "patch_size": 200                                  //patch size to inference
  }
}
```

## Acknowledgement
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m)
- [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

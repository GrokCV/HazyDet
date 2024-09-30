# HazyDet: Open-source Benchmark for Drone-View Object Detection with Depth-cues in Hazy Scenes
This repository is the official implementation of HazyDet

- [HazyDet](#HazyDet)

- [Leadboard and Model Zoo](#leadboard-and-model-zoo)
   - [Detectors](#detectors)
   - [Dehazing](#dehazing)

- [DeCoDet](#DeCoDet)
    - [Installation](#Installation)
        - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
        - [Step 2: Install PyTorch](#step-2-install-pytorch)
        - [Step 3: Install OpenMMLab 2.x Codebases](#step-3-install-openmmlab-2x-codebases)
        - [Step 4: Install `HazyDet`](#step-4-install-HazyDet)
    - [Training](#training)
    - [Inference](#inference)








## HazyDet

![HazyDet](./docs/dataset_samples.jpg)
You can download our HazyDet-365K dataset from [here](https://pan.baidu.com/s/1KKWqTbG1oBAdlIZrTzTceQ?pwd=grok).<br>
For both training and inference, the following dataset structure is required:

```
HazyDet-365K
|-- train
    |-- clean images
    |-- hazy images
    |-- labels
|-- val
    |-- clean images
    |-- hazy images
    |-- labels
|-- test
    |-- clean images
    |-- hazy images
    |-- labels
|-- RDDTS
    |-- hazy images
    |-- labels
```

**Note: Both passwords for BaiduYun and OneDrive is `grok`**.



## Leadboard and Model Zoo

All the weight files in the model zoo can be accessed on [Baidu Cloud](https://pan.baidu.com/s/1EEX_934Q421RkHCx53akJQ?pwd=grok) and [OneDrive](https:).

### Detectors


<table>
    <tr>
        <td>Model</td>
        <td>Backbone</td> <!-- 新增列 -->
        <td>#Params (M)</td>
        <td>GFLOPs</td>
        <td>Test-set</td>
        <td>RDDTS</td>
        <td>Config</td>
    </tr>
    <tr>
        <td>YOLOv3</td>
        <td>Darknet53</td> <!-- 新增内容 -->
        <td>61.63</td>
        <td>20.19</td>
        <td>35.0</td>
        <td>19.2</td>
        <td><a href="config/yolov3/yolov3_d53_320_273e_coco.py">config</a></td>
    </tr>
    <tr>
        <td>GFL</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.26</td>
        <td>198.65</td>
        <td>36.8</td>
        <td>13.9</td>
        <td><a href="config/gfl/gfl_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>YOLOX</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>8.94</td>
        <td>13.32</td>
        <td>42.3</td>
        <td>24.7</td>
        <td><a href="config/yolox/yolox_s_8x8_300e_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>RepPoints</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>36.83</td>
        <td>184.32</td>
        <td>43.8</td>
        <td>21.3</td>
        <td><a href="config/reppoints/reppoints_moment_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>FCOS</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.11</td>
        <td>191.48</td>
        <td>45.9</td>
        <td>22.8</td>
        <td><a href="config/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Centernet</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.11</td>
        <td>191.49</td>
        <td>47.2</td>
        <td>23.8</td>
        <td><a href="config/centernet/centernet_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>ATTS</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.12</td>
        <td>195.58</td>
        <td>50.4</td>
        <td>25.1</td>
        <td><a href="config/atts/atts_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>DDOD</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.20</td>
        <td>173.05</td>
        <td>50.7</td>
        <td><u>26.1</u></td>
        <td><a href="config/ddod/ddod_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>VFNet</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.89</td>
        <td>187.39</td>
        <td>51.1</td>
        <td>25.6</td>
        <td><a href="config/vfnet/vfnet_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>TOOD</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>32.02</td>
        <td>192.51</td>
        <td>51.4</td>
        <td>25.8</td>
        <td><a href="config/tood/tood_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Sparse RCNN</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>108.54</td>
        <td>147.45</td>
        <td>27.7</td>
        <td>10.4</td>
        <td><a href="config/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Dynamic RCNN</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>41.35</td>
        <td>201.72</td>
        <td>47.6</td>
        <td>22.5</td>
        <td><a href="config/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Faster RCNN</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>41.35</td>
        <td>201.72</td>
        <td>48.7</td>
        <td>23.6</td>
        <td><a href="config/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Libra RCNN</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>41.62</td>
        <td>209.92</td>
        <td>49.0</td>
        <td>23.7</td>
        <td><a href="config/libra_rcnn/libra_rcnn_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Grid RCNN</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>64.46</td>
        <td>317.44</td>
        <td>50.5</td>
        <td>25.2</td>
        <td><a href="config/grid_rcnn/grid_rcnn_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Cascade RCNN</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>69.15</td>
        <td>230.40</td>
        <td><u>51.6</u></td>
        <td>26.0</td>
        <td><a href="config/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Conditional DETR</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>43.55</td>
        <td>94.17</td>
        <td>30.5</td>
        <td>11.7</td>
        <td><a href="config/conditional_detr/conditional_detr_r50_8x2_150e_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>DAB DETR</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>43.70</td>
        <td>97.02</td>
        <td>31.3</td>
        <td>11.7</td>
        <td><a href="config/dab_detr/dab_detr_r50_8x2_50e_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>Deform DETR</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>40.01</td>
        <td>192.51</td>
        <td><b>51.9</b></td>
        <td><b>26.5</b></td>
        <td><a href="config/deform_detr/deform_detr_r50_16x2_50e_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
    <tr>
        <td>FCOS-DeCoDet</td>
        <td>ResNet50</td> <!-- 新增内容 -->
        <td>34.61</td>
        <td>249.91</td>
        <td>47.4</td>
        <td>24.3</td>
        <td><a href="config/fcos_decodet/fcos_decodet_r50_fpn_1x_coco.py">config</a></td> <!-- 新增链接 -->
    </tr>
</table>

### Dehazing


## DeCoDet
![HazyDet-365K](./docs/network.jpg)

### Installation

#### Step 1: Create a conda 

```shell
$ conda create --name HazyDet python=3.9
$ source activate HazyDet
```

#### Step 2: Install PyTorch

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Step 3: Install OpenMMLab 2.x Codebases

```shell
# openmmlab codebases
pip install -U openmim --no-input
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmsegmentation>=1.0.0" "mmrotate>=1.0.0rc1" mmyolo "mmpretrain>=1.0.0rc7" 'mmagic'
# other dependencies
pip install -U ninja scikit-image --no-input
```

#### Step 4: Install `HazyDet`

```shell
python setup.py develop
```

**Note**: make sure you have `cd` to the root directory of `HazyDet`

```shell
$ git clone git@github.com:GrokCV/HazyDet.git
$ cd HazyDet
```

### Training
```shell
 $ python tools/train_det.py configs/DeCoDet/DeCoDet_r50_1x_hazydet.py
```         


### Inference
```shell
$ python tools/test.py configs/DeCoDet/DeCoDet_r50_1x_hazydet365k.py weights/fcos_DeCoDet_r50_1x_hazydet.pth
```

We released our [checkpoint](https://pan.baidu.com/s/1EEX_934Q421RkHCx53akJQ?pwd=grok) on HazyDet <br>

### Depth Maps

The depth map required for training can be obtained through [Metic3D](https://github.com/YvanYin/Metric3D). They can also be acquired through other depth estimation models.<br>
## Acknowledgement
We are grateful to the Tianjin Key Laboratory of Visual Computing and Intelligent Perception (VCIP) for providing essential resources. Our sincere appreciation goes to Professor Pengfei Zhu and the dedicated AISKYEYE team at Tianjin University for their invaluable support with data, which has been crucial to our research efforts. We also deeply thank Xianghui Li, Yuxin Feng, and other researchers for granting us access to their datasets, significantly advancing and promoting our work in this field. Additionally, our thanks extend to [Metric3D](https://github.com/YvanYin/Metric3D) for its contributions to the methodology presented in this article. 


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{feng2024HazyDet,
	title={HazyDet: Open-source Benchmark for Drone-view Object Detection with Depth-cues in Hazy Scenes}, 
	author={Changfeng, Feng and Zhenyuan, Chen and Renke, Kou and Guangwei, Gao and Chunping, Wang and Xiang, Li and Xiangbo, Shu and Yimian, Dai and Qiang, Fu and Jian, Yang},
	year={2024},
	journal={arXiv},
}

@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7380--7399},
  year={2021},
  publisher={IEEE}
}
```
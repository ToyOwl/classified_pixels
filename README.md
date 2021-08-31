Implementation of DL architectures for semantic segmentation of images and point clouds: vanilla U-Net [1], DeepLabV3 [2], DeepLabV3+ [3], RandLaNet [4]

Train scripts ./segmentation_models/train_vaihingen.py and ./segmentation_models/ train_dales.py for [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/3d-semantic-labeling/) and [Dales](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) datasets.

Inference scripts ./inference/inference_vaihingen.py and ./inference/inference_dales.py 

### Segmentation examples:

* DaLes Dataset semantic segmentation with RandLaNet-5

excellent: 

![](./pics/pic_01.png)
![](./pics/pic_02.png)
![](./pics/pic_03.png)
![](./pics/pic_04.png)
![](./pics/pic_05.png)


and bad:

![](./pics/pic_06.png)
![](./pics/pic_07.png)
![](./pics/pic_08.png)

here blue - GLO,  green - vegetation, red - buildings, magenta - clutter.

Metrics: 0.80 mIoU,  GLO 0.84 IoU, vegetation 0.79 IoU, buildings 0.75 IoU, clutter 0.75 IoU. 

* Buildings detection (one class segmentation) for Massachusetts datasets with DeepLabV3+ (ResNet-152 backbone) 0.85 IoU

![](./pics/pic_09.png)
![](./pics/pic_10.png)
![](./pics/pic_11.png)


* Real life aerial pics segmentation examples: 

![](./pics/pic_12.png)
![](./pics/pic_15.png)


Trained models [RandLaNet and DeepLabV3+ models](https://disk.yandex.ru/d/p7zLEl2ruNXQ3w)

[DaLes classified clouds](https://disk.yandex.ru/d/ZBZGTQA78s7jcA)

[1] Olaf Ronneberger and Philipp Fischer and Thomas Brox. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[2] Liang-Chieh Chen and George Papandreou and Florian Schroff and Hartwig Adam. [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[3] Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[4]Qingyong Hu and Bo Yang and Linhai Xie and Stefano Rosa and Yulan Guo and Zhihua Wang and Niki Trigoni and Andrew Markham. [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236)

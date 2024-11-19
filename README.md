# DA-Faster

## 环境配置

1. 该项目基于[detectron-v0.6](https://github.com/facebookresearch/detectron2/tree/v0.6),在使用前确保自己已经配置好了环境。
2. 数据集注册在`/da_faster/data/register`中，使用COCO格式，需要使用者自行修改数据集路径。目前只提供cityscapes->foggy_cityscapes场景，其余场景自行注册。

## 预训练权重

在主目录下创建`weights`,然后从[这里](https://drive.google.com/file/d/1wNIjtKiqdUINbTUVtzjSkJ14PpR2h8_i/view?usp=sharing)下载vgg-16的预训练权重，并放入`weights`文件夹中。

## 用法

* da_traing:

```
python ./train_net_da.py --config-file ./configs/faster-rcnn-vgg16-CrossCityscapes.yaml --num-gpus 4
```

* source_only_training:

```
python ./train_net_so.py --config-file ./configs/faster-rcnn-vgg16-CrossCityscapes_so.yaml --num-gpus 4
```

注意：想要更改训练配置，请在`config/`目录下更改训练config文件，或者在`da_faster/config.py`中更改。（cfg中的参数更改也支持命令行传参）

## 联系方式

如果有任何问题，欢迎联系我：<suojinhui@gmail.com>

-----前言-----
本源码为我带领团队参加2019 CCF BDCI 第一赛道《视频侵权检测》算法竞赛获得一等奖时所使用的部分源码，包括视频抽帧、图像卷积特征提取、SPOC特征聚合等方法的实现。由于产品化的原因，后续的特征匹配和视频匹配算法均使用C/C++编写，且不便公开，敬请谅解。

-----使用说明-----
1）由于用到的几个caffe模型存档文件SIZE较大，在本代码中已经删除掉，删除模型的路径记录如下：
	caffe/examples/cifar10/cifar10_quick_iter_4000.caffemodel
	caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel
	caffe/models/hvr2/hvr_alexnet_train_iter_5000.caffemodel（自）
	caffe/models/vgg16/VGG_ILSVRC_16_layers.caffemodel

2) edit ~/.profile, add a line as below at the end of the file: 
    export PYTHONPATH=.../lib:$PYTHONPATH	
	
3）程序说明
	my_video.py 视频抽帧
	my_infer.py 卷积特征提取
	my_eval.py  特征比对
	spoc_train.py  	训练集特征聚合
	spoc_test.py	测试集特征聚合


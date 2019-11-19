# 复现细节
* 第一步，使用YOLO切割原本数据，切割好的数据已放置在data_crop文件夹下
* 第二步，将图片数据使用read_data.py转化为pkl文件方便文件读取，转化后的数据已放置在data_pkl下
* 第三步，运行train_basic_model_fast.py十次，训练十个模型，每次训练需要更改随机数种子和val_number以确保bagging的随机性，val_number个人一般是选取800-1200之间的数值
* 第四步，每次选取出验证集accuracy最优的模型，删除其余的模型
* 第五步，使用predict_basic_model.py进行推断，推断前需要更改模型路径和结果保存路径
* 第六步，得到十个结果文件后，进入结果文件夹，使用merge.py对结果进行融合
* 基于train_att_model_fast.py重复3-6步，得到attention模型的融合结果
* 基于train_deep_model_fast.py重复3-6步，得到deep模型的融合结果
* 对三种模型的三个融合结果，进入merge_result文件夹，运行merge.py进行融合得到最终的结果

## 运行环境
```shell script
Python 3.6.5
Keras 2.1.5
tensorflow 1.6.0
Pytorch 1.2.0
torchvision 0.4.0
tqdm 4.35.0
cuda 9.0.176
cudnn 7.6.2
```

## 主要方案进化曲线
* 1.未切割嘴巴 lipNet ===> A榜 0.12
* 2.切割嘴巴 D3D ===> A榜 0.5
* 3.3D卷积 + ResNet + RNN 单模型最优 A榜 0.735
* 4.3D卷积 + ResNet + RNN 五模型融合 A榜 0.77
* 5.3D + ResNet + RNN / Deep 3D + ResNet + RNN / 3D + SENet + RNN\
3种模型，每个模型对数据随机抽样训练出10个模型, 每一种模型，10个ensemble之后分数大概在0.78左右
* 6.三种模型ensemble的结果再进行merge ==> 0.785


## 处理流程
#### 数据处理
* 1.对训练测试数据的随机抽取1000个文件夹，每个文件夹抽取1张图片，共1000张图使用label image对嘴巴进行标注
* 2.用标注的1000张图训练一个YOLO V3的目标检测网络，并用该网络处理原有数据
* 3.标注的1000图像放在YOLOv3/data/目录下
* 4.训练数据放在YOLOv3/lip_train/下，测试数据放在YOLOv3/lip_test/下
* 5.将YOLO权重文件放在YOLOv3/logs/下
* 6.按照如下脚本进行处理，最终处理之后的数据图像会将原图覆盖掉，未检测到嘴巴的图像是数据中的异常值，需要手动删除
```shell script
cd YOLOv3/
python yolo_video_test.py
python yolo_video_train.py
```

#### 验证集划分
* 1.原始数据放置在data_original下，包含训练数据data_original/lip_train/*,测试数据data_original/lip_test/*,label数据lip_train.txt
* 2.YOLO处理好的切割数据移动到data_crop下，lip_train/*以及lip_test/*
* 3.使用cut_data.py划分验证集训练集，主要有两种方式，一是随机采样，一种每种类别均匀采样
* 4.在文件中有三个参数需要设置, uniform_get,设置为True or False
* 5.val_pre_number为每个类别取多少个样本，一般是3，对应均匀取样
* 6.val_number为总样本每个类别取多少个，一般为1000，对应随机采样
* 7.每次训练签需要运行cut_data.py，对样本采样，以保证模型的多样性
```shell script
python cut_data.py
```

#### 数据读取
* 1.数据读取主要是基于torch的dataloader
* 2.训练数据读取LipReadDataTrain.py
* 3.测试数据读取LipReadDataTest.py
* 4.数据读取主要采取了两种数据增强策略：图片放大1.2倍后随机裁剪出112*112的区域，整体镜像

#### 模型训练
* 1.训练基本模型，方案为3D + 2D ResNet + RNN 
```shell script
python train_basic_model.py
```
* 2.训练注意力模型，方案为3D + 2D SENet + RNN 
```shell script
python train_att_model.py
```
* 3.训练Deep 3D模型，方案为3D + 2D ResNet + RNN 
```shell script
python train_deep_model.py
```

#### 数据快速读取
* 1.dataloder读取数据过于缓慢，每次都要从磁盘中读图片，导致训练时间过长,后续训练均使用快速训练的方式
* 2.将图片数据一次性读取，写入pkl文件，训练时全部读取到内存,然后每次从内存取数据
* 3.运行read_data.py将文件一次性读取，写入好的文件放在data_pkl/*中，分别为data_train.pkl,data_test.pkl
* 4.data_train.pkl中为训练数据的图片和label index, data_test.pkl中为图片数据和文件名
* 5.验证集划分的方式直接使用index进行划分,这样训练更高效
```shell script
python read_data.py
```

#### 模型快速训练
每次模型训练，为保证bagging的多样性，需要重新设置随机数种子和val_number,val_number取过2000,1600,1400,1200,1000,900,800等
* 1.训练基本模型，方案为3D + 2D ResNet + RNN 
```shell script
python train_basic_model_fast.py
```
* 2.训练注意力模型，方案为3D + 2D SENet + RNN 
```shell script
python train_att_model_fast.py
```
* 3.训练Deep 3D模型，方案为3D + 2D ResNet + RNN 
```shell script
python train_deep_model_fast.py
```

#### 模型推断
* 1.基本模型预测，更改模型路径和结果保存路径，运行脚本
```shell script
python predict_basic_model.py
```
* 2.注意力模型预测，更改模型路径和结果保存路径，运行脚本
```shell script
python predict_att_model.py
```
* 3.Deep模型预测，更改模型路径和结果保存路径，运行脚本 
```shell script
python predict_deep_model.py
```

## 整体流程
1. 三种模型，每种模型每次训练时设置不一样的val_number,和随机数种子
2. 训练完成后，保存验证集准确率最优的模型，删除其他模型
3. 每种模型训练大概10个模型进行推断
4. 对每种模型的结果进行融合
```shell script
python predict_deep_model.py
```
## 结果处理
原始的一些模型推断结果已放入results/*下的各个文件夹里

进行模型推断，需要把权重文件夹中的权重放入weights/*下的各个文件夹中，更改路径进行推断
```shell script
cd results/attention/
python merge.py
cp attention_submit.csv ../../merge_result

cd results/basic/
python merge.py
cp basic_submit.csv ../../merge_result

cd results/deep/
python merge.py
cp deep_submit.csv ../../merge_result
```
这样一共得到三份结果文件，并将这三份结果文件转移到merge_result中

这三份文件基本预测准确率在A榜0.78+左右的样子
```shell script
cd merge_result
python merge.py
```
将这三分结果做最终的融合，得到最终结果fusion_submit.csv

## 一点说明
A榜最高的0.791的结果是最后尝试融合了十多份结果比较好的成绩得到的最终结果，这份结果放置在final_results中
```shell script
cd final_results
python merge.py
```
由于个人操作的原因，有一些模型文件有缺失，但是不影响最终的分数，按照目前ensemble的思路达到现在的效果问题不大


## Reference
1. YOLOv3: https://github.com/qqwweee/keras-yolo3
2. lipNet: https://github.com/Leviclt/lip_reading_demo_net
3. D3D: https://github.com/NirHeaven/D3D
4. D3D: https://github.com/Fengdalu/Lipreading-DenseNet3D
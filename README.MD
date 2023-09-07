#1.数据集说明
mnist：https://zhuanlan.zhihu.com/p/547147237?utm_id=0

#2.文件说明：
- checkpoints：模型文件存储位置
- data：数据集存储位置
- model：模型代码
- train.py: 训练代码
- predict.py：预测代码

#2.使用流程：
- 运行train.py进行训练
```
右键run，会使用默认值运行。

或者命令行传参运行：
python train.py --gpu 0 --epochs 50 --batch_size 64 --lr 0.1 --dataset_path ./data/
```
- 运行predict.py进行单个图片的预测
```
右键run，会使用默认值运行。

或者命令行传参运行：
python predict.py --gpu 0 --model_savepath ./checkpoints/mnist-lenet5.pth --dataset_path ./data/
```
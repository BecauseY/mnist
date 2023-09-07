import argparse
import os

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from models.lenet5 import lenet5


def predict(args):
    # 加载模型
    model = lenet5(input_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model_savepath), strict=True)
    model.eval()  # switch to eval status

    #加载数据
    train_datas = datasets.MNIST(root=args.dataset_path, train=True, download=True)
    # 这里从训练集里面取出来几张图片进行预测，要将图片转化为tensor，并转化为模型需要的输入格式
    train_data = train_datas.data
    train_label = train_datas.targets
    for i in range(9):
        data = train_data[i]
        ture_label = train_label[i]
        data = data.numpy()

        # 画图
        plt.imshow(data)
        plt.show()

        # 转化为tensor
        data = torch.from_numpy(data)
        # 模型期望一个 4D 张量作为输入（批量大小 x 通道 x 高度 x 宽度）
        data = data.unsqueeze(0).unsqueeze(0).float().to(device)

        # 单个图片输入模型
        y_predict = model(data)
        y_predict = torch.argmax(y_predict, dim=1)
        print(f"{i}:    true label:{ture_label}     predict label:{y_predict.cpu().numpy()[0]}")

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Predict ')

    parser.add_argument('--gpu', default='0', help='Which gpu to use, default:0')
    parser.add_argument('--model_savepath', default='./checkpoints/mnist-lenet5.pth', help='Which model to use')
    parser.add_argument('--dataset_path', default='./data/', help='Place to load dataset, default: ./data/')

    args = parser.parse_args()

    # gpu若存在则用gpu，不然用cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict(args)
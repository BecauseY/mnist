import argparse
import os

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from models.lenet5 import lenet5


def train_one_epoch(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    y_true = []
    y_predict = []
    model.train()
    for (batch_x, batch_y) in tqdm(data_loader, desc='Training'):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        output = torch.argmax(output, dim=1)
        y_true.append(batch_y)
        y_predict.append(output)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    return running_loss / len(data_loader), accuracy_score(y_true.cpu(), y_predict.cpu())


def evaluate(data_loader, model, device):
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    with torch.no_grad():  # 在进行预测时禁用不必要的梯度计算以节省计算资源,是一种优化方法,不写也可以运行
        for (batch_x, batch_y) in tqdm(data_loader, desc='Testing'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_predict = model(batch_x)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_true.append(batch_y)
            y_predict.append(batch_y_predict)

        y_true = torch.cat(y_true, 0)
        y_predict = torch.cat(y_predict, 0)

    return accuracy_score(y_true.cpu(), y_predict.cpu())


def train(args):
    # 加载数据集,这里下载pytorch框架自带的数据集mnist，后续需要自己写代码加载数据
    # 注意需要把数据格式变为tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_datas = datasets.MNIST(root=args.dataset_path, train=True, download=True, transform=transform)
    test_datas = datasets.MNIST(root=args.dataset_path, train=False, download=True, transform=transform)

    # 加载loader
    train_loader = DataLoader(train_datas, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_datas, batch_size=args.batch_size, shuffle=False)

    # 配置网络
    model = lenet5(input_channels=1, num_classes=10).to(device)

    # 配置优化器，这里用的是SGD，也可用adam等其他的优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 配置损失函数
    criterion = torch.nn.CrossEntropyLoss()
    model_savepath = "./checkpoints/mnist-lenet5.pth"

    if os.path.exists(model_savepath):
        # 如果存在模型，读取一下并测试性能
        print("model exist")
        print(f"Load model from : {model_savepath}")
        model.load_state_dict(torch.load(model_savepath), strict=True)
        test_acc = evaluate(test_loader, model, device)
        print(f"Test Acc: {test_acc}")

    else:
        # 如果没有模型，则重新训练一个模型
        print("model not exist")
        print(f"Start training for {args.epochs} epochs")
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device)
            test_acc = evaluate(test_loader, model, device)
            print(f"# EPOCH {epoch}  Train Loss: {train_loss} Train Acc: {train_acc} Test Acc: {test_acc}\n")

            scheduler.step()
            torch.save(model.state_dict(), model_savepath)


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Train a model ')

    parser.add_argument('--gpu', default='0', help='Which gpu to use, default:0')
    parser.add_argument('--epochs', default=50, help='Number of epochs to train backdoor model, default: 10')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate of the model, default: 0.1')
    parser.add_argument('--dataset_path', default='./data/', help='Place to load dataset, default: ./data/')

    args = parser.parse_args()

    # gpu若存在则用gpu，不然用cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(args)

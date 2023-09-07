import torch.nn as nn

#原lenet5输入是1*32*32，这是一个变种
#当数据集是mnist时，input_channels = 1 ,修改模型结构，让fc1_input_features为16*4*4
#当数据集是cifar10时，input_channels = 3 ,不修改网络结构，fc1_input_features为16*5*5，和原始网络机构的一样
#所以用mnist和cifar10时，不用resize
class lenet5(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(lenet5, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        fc1_input_features = 16*5*5 if input_channels == 3 else 16*4*4
        self.fc_layers = nn.Sequential(
            nn.Linear(fc1_input_features, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

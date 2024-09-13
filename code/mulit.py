import torch
import torch.nn as nn
# 定义一个单独的分支，用于处理一个视图的8张图片
class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()
        # 第一个卷积层，输入是8×128×128×1，输出是8×64×64×32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2,padding=1)
        # 第二个卷积层，输入是8×64×64×32，输出是8×64×64×32
        self.conv2_dw = nn.Conv2d(32, 32, kernel_size=3, groups=32,padding=1)
        self.conv2_pw = nn.Conv2d(32, 64, kernel_size=1)
        # 第三个卷积层，输入是8×64×64×64，输出是8×32×32×64
        self.conv3_dw = nn.Conv2d(64, 64, kernel_size=3, stride=2, groups=64,padding=1)
        self.conv3_pw = nn.Conv2d(64, 128, kernel_size=1)
        # 第四个卷积层，输入是8×32×32×128，输出是8×16×16×128
        self.conv4_dw = nn.Conv2d(128, 128, kernel_size=3, stride=2, groups=128,padding=1)
        self.conv4_pw = nn.Conv2d(128, 128, kernel_size=1)
        # 第五个卷积层，输入是8×16×16×128，输出是8×8×8×128
        self.conv5_dw = nn.Conv2d(128, 128, kernel_size=3, stride=2, groups=128,padding=1)
        self.conv5_pw = nn.Conv2d(128, 128, kernel_size=1)

    def forward(self, x):
        # x的形状是batch_size x num_images_per_view x num_channels x height x width
        batch_size = x.shape[0]
        num_images_per_view = x.shape[1]
        x = x.reshape(batch_size * num_images_per_view,
                      *x.shape[2:])  # 转换形状为 (batch_size * num_images_per_view) x num_channels x height x width
        x = self.conv1(x)  # x的形状是(batch_size *8)×32×64×64
        x = self.conv2_dw(x)  # x的形状是(batch_size *8)×32×64×64
        x = self.conv2_pw(x)  # x的形状是(batch_size *8)×64×64×64
        x = self.conv3_dw(x)  # x的形状是(batch_size *8)×64×32×32
        x = self.conv3_pw(x)  # x的形状是(batch_size *8)×128×32×32
        x = self.conv4_dw(x)  # x的形状是(batch_size *8)×128×16×16
        x = self.conv4_pw(x)  # x的形状是(batch_size *8)×128×16×16
        x = self.conv5_dw(x)  # x的形状是(batch_size *8)×128×8×8
        x = self.conv5_pw(x)  # x的形状是(batch_size *8)×128×8×8
        x=x.view(batch_size, num_images_per_view, *x.shape[1:])
        return x


# 定义一个多分支网络，用于处理五个视图的数据
class MultiBranch(nn.Module):
    def __init__(self):
        super(MultiBranch, self).__init__()
        # 定义五个分支，每个分支处理一个视图的数据
        self.branch1 = Branch()
        self.branch2 = Branch()
        self.branch3 = Branch()
        self.branch4 = Branch()
        self.branch5 = Branch()
        # 定义两个全连接层，用于合并五个分支的特征并进行分类
        self.fc1 = nn.Linear(81920, 5120)
        self.fc2 = nn.Linear(5120, 64)
        # 定义一个分类器，用于输出二分类结果
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        # x的形状是batch_size x 5 x 8 x 1 x 128 x 128，表示batch_size个样本，每个样本有5个视图，每个视图有8张图片，每张图片有一个通道和大小为128x12
        # 将x拆分为五个视图的数据，并分别送入对应的分支中
        x1, x2, x3, x4, x5 = x.split(1, dim=1)
        x1 = x1.squeeze(dim=1)  # x1的形状是batch_size x 8 x 1 x 128 x 128
        x1 = self.branch1(x1)  # x1的形状是batch_size x 8 x 1 x 128 x 128
        x2 = x2.squeeze(dim=1)  # x2的形状是batch_size x 8 x 1 x 128 x 128
        x2 = self.branch2(x2)  # x2的形状是batch_size x 8 x 1 x 128 x 128
        x3 = x3.squeeze(dim=1)  # x3的形状是batch_size x 8 x 1 x 128 x 128
        x3 = self.branch3(x3)  # x3的形状是batch_size x 8 x 1 x 128 x 128
        x4 = x4.squeeze(dim=1)  # x4的形状是batch_size x 8 x 1 x 128 x 128
        x4 = self.branch4(x4)  # x4的形状是batch_size x 8 x 1 x 128 x 128
        x5 = x5.squeeze(dim=1)  # x5的形状是batch_size x 8 x 1 x 128 x 128
        x5 = self.branch5(x5)  # x5的形状是batch_size x 8 x 1 x 128 x 128
        # 将五个分支的输出展平并拼接起来，得到一个40960维的向量
        x1 = torch.flatten(x1, start_dim=1)  # x1的形状是8×8192*8
        x2 = torch.flatten(x2, start_dim=1)  # x2的形状是8×8192*8
        x3 = torch.flatten(x3, start_dim=1)  # x3的形状是8×8192*8
        x4 = torch.flatten(x4, start_dim=1)  # x4的形状是8×8192*8
        x5 = torch.flatten(x5, start_dim=1)  # x5的形状是8×8192*8
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)  # x的形状是8×40960
        # 送入两个全连接层，得到一个128维的特征向量
        x = self.fc1(x)  # x的形状是8×5120
        x = self.fc2(x)  # x的形状是8×128

        # 送入分类器，得到一个二分类结果
        out = self.classifier(x)  # out的形状是8×1
        # out = torch.softmax(out)  # out的形状是8×1，值在0到1之
        return out



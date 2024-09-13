import os
from mulitviewdataset import MultiViewDataset
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
from mulit import MultiBranch
import matplotlib.pyplot as plt


# 调整学习率的策略
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 10))
    # print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程
def train(model, device, train_loader, val_loader, optimizer, EPOCHS, criterion, save_name):
    best_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        sum_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = Variable(data).to(device), Variable(label).to(device)
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % int(BATCH_SIZE / 10) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()))
        ave_loss = sum_loss / len(train_loader)

        # 保存模型
        if ave_loss < best_loss:
            best_loss = ave_loss
            torch.save(model, save_name)
        train_loss_list.append(ave_loss)
        print('epoch:{},loss:{}'.format(epoch, ave_loss))

        # val(model, DEVICE, val_loader)

        # 验证模型
        model.eval()
        test_loss = 0
        correct = 0
        total_num = len(val_loader.dataset)
        with torch.no_grad():
            for data, label in val_loader:
                data, label = Variable(data).to(device), Variable(label).to(device)
                out = model(data)
                _, pred = torch.max(out.data, 1)
                correct += torch.sum(pred == label)
                loss = criterion(out, label)
                print_loss = loss.data.item()
                test_loss += print_loss
            correct = correct.item()
            acc = correct / total_num
            avgloss = test_loss / len(val_loader)
            val_loss_list.append(ave_loss)
            val_acc_list.append(acc)
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                avgloss, correct, len(val_loader.dataset), 100 * acc))

    x = range(1, EPOCHS + 1)

    y1 = train_loss_list
    y2 = val_loss_list
    y3 = val_acc_list

    plt.subplot(3, 1, 1)
    plt.plot(x, y1, 'o-')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')

    plt.subplot(3, 1, 2)
    plt.plot(x, y2, '-')
    plt.title('val loss vs. epoches')
    plt.ylabel('val loss')

    plt.subplot(3, 1, 3)
    plt.plot(x, y3, '-')
    plt.title('val acc vs. epoches')
    plt.ylabel('val acc')

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.5)
    name = save_name.split("/")[-1].split(".pth")[0] + ".jpg"
    picture_save_path = "H:\二尖瓣项目\mulit_loss"
    plt.savefig(os.path.join(picture_save_path, name))


# 训练
if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_path = r"H:\二尖瓣项目\dataset\mulit_dataset\train"
    val_path = r"H:\二尖瓣项目\dataset\mulit_dataset\val"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # 读取数据
    dataset_train = MultiViewDataset(train_path, transform=transform)
    dataset_val = MultiViewDataset(val_path, transform=transform)
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiBranch()
    model.to(DEVICE)

    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                             pin_memory=True)

    modellr = 1 * 1e-3
    optimizer = optim.Adam(model.parameters(), lr=modellr)
    EPOCHS = 100
    criterion = nn.CrossEntropyLoss()
    save_name = "mulit_model/model_1_e100_lr=0.001.pth"
    train(model, DEVICE, train_loader, val_loader, optimizer, EPOCHS, criterion, save_name)

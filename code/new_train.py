import os
from new_whole_dataset import MVP_whole_Dataset_new
import torch.optim as optim
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from new_resnet18 import Resnet18
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 10))
    # print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程
def train(model, device, train_loader, val_loader, optimizer, EPOCHS, criterion, save_name):
    # print(train_path,val_path)
    # print("modellr",modellr)
    best_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        sum_loss = 0
        for batch_idx, (data, view, target) in enumerate(train_loader):
            data, view, target = data.to(device), view.to(device), target.to(device)
            output = model(data, view)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % 32 == 0:
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
        # print(total_num, len(val_loader))
        with torch.no_grad():
            for data, view,target in val_loader:
                data, view, target = Variable(data).to(device), Variable(view).to(device), Variable(target).to(device)
                output = model(data, view)
                loss = criterion(output, target)
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == target)
                print_loss = loss.data.item()
                test_loss += print_loss
            correct = correct.data.item()
            acc = correct / total_num
            avgloss = test_loss / len(val_loader)
            val_loss_list.append(ave_loss)
            val_acc_list.append(acc)
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                avgloss, correct, len(val_loader.dataset), 100 * acc))
    x1 = range(1, EPOCHS + 1)
    x2 = range(1, EPOCHS + 1)
    x3 = range(1, EPOCHS + 1)
    y1 = train_loss_list
    y2 = val_loss_list
    y3 = val_acc_list
    plt.subplot(3, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.subplot(3, 1, 2)
    plt.plot(x2, y2, '-')
    plt.title('val loss vs. epoches')
    plt.ylabel('val loss')
    plt.subplot(3, 1, 3)
    plt.plot(x3, y3, '-')
    plt.title('val acc vs. epoches')
    plt.ylabel('val acc')

    plt.subplots_adjust(hspace=0.5)
    name = save_name.split("/")[-1].split(".pth")[0] + ".jpg"
    picture_save_path = r"H:\二尖瓣项目\new_whole_model_loss"
    plt.savefig(os.path.join(picture_save_path, name))


# 训练
if __name__ == "__main__":
    train_path = r"H:\二尖瓣项目\dataset\whole_data\train"
    val_path = r"H:\二尖瓣项目\dataset\whole_data\val"
    # 读取数据
    dataset_train = MVP_whole_Dataset_new(train_path, 224, mode="train")
    dataset_val = MVP_whole_Dataset_new(val_path, 224, mode="val")

    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Resnet18(2)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                             pin_memory=True)

    modellr = 9 * 1e-3
    optimizer = optim.Adam(model.parameters(), lr=modellr)

    EPOCHS = 100
    save_name = "new_whole_model/best_model_7_e100_lr=0.009.pth"
    train(model, DEVICE, train_loader, val_loader, optimizer, EPOCHS, criterion, save_name)

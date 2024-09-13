import os
from whole_dataset import MVP_whole_Dataset
from Dataset import MVP_Dataset
import torch.optim as optim
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from resnet18 import Resnet18
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 10))
    # print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 验证过程
def val(model, device, test_loader, mode):
    # 验证模型
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    # print(total_num, len(val_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(val_loader)
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(mode,
                                                                                   avgloss, correct,
                                                                                   len(test_loader.dataset), 100 * acc))
        return avgloss, acc


# 定义训练过程
def train(model, device, train_loader, val_loader, optimizer, EPOCHS, criterion, save_name):
    # print(train_path,val_path)
    # print("modellr",modellr)
    best_loss = float('inf')
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        model.train()
        sum_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            # print(batch_idx)
            if (batch_idx + 1) % (BATCH_SIZE//10) == 0:
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
        _, train_acc = val(model, device, train_loader, "train")
        val_loss, val_acc = val(model, device, val_loader, "val")
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    x = range(1, EPOCHS + 1)
    y1 = train_loss_list
    y2 = train_acc_list
    y3 = val_loss_list
    y4 = val_acc_list

    # plt.subplots_adjust(hspace=0.5)
    save_name_loss = save_name.split("/")[-1].split(".pth")[0] + "_" + "loss.jpg"
    save_name_acc = save_name.split("/")[-1].split(".pth")[0] + "_" + "acc.jpg"
    picture_save_path = r"H:\二尖瓣项目\3 CLASS Final\whole_loss"

    # Plot Train Loss and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, 'b', marker='.', label='Train Loss')
    plt.plot(x, y3, 'orange', marker='.', label='Validation Loss')
    plt.title('Train Loss and Validation Loss vs. Epoches')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(picture_save_path, save_name_loss))

    # Plot Train Accuracy and Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(x, y2, 'b', marker='.', label='Train Accuracy')
    plt.plot(x, y4, 'orange', marker='.', label='Validation Accuracy')
    plt.title('Train Accuracy and Validation Accuracy vs. Epoches')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(picture_save_path, save_name_acc))


# 训练
if __name__ == "__main__":
    train_path = r"H:\二尖瓣项目\dataset\final 3 class dataset\train"
    val_path = r"H:\二尖瓣项目\dataset\final 3 class dataset\val"
    save_name = "3 CLASS Final/whole_model/best_model_5_e50_lr=0.0005.pth"
    # 读取数据
    dataset_train = MVP_whole_Dataset(train_path, 224, mode="train")
    dataset_val = MVP_whole_Dataset(val_path, 224, mode="val")

    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Resnet18(3)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                             pin_memory=True)
    modellr = 5 * 1e-4
    optimizer = optim.Adam(model.parameters(), lr=modellr)
    EPOCHS = 50
    train(model, DEVICE, train_loader, val_loader, optimizer, EPOCHS, criterion, save_name)

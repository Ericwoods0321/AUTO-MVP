import torch
from torch.utils.data import DataLoader
from Dataset import MVP_Dataset
from whole_dataset import MVP_whole_Dataset

from matrix import ConfusionMatrix
import time
import os


def main():
    # Step 0:查看torch版本、设置device
    # print(torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载测试集
    test_path = r"H:\二尖瓣项目\dataset\final 3 class dataset\val"
    dataset_test = MVP_whole_Dataset(test_path, 224, mode="test")
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    # Step 3：加载训练好的权重
    model = torch.load(r"H:\二尖瓣项目\3 CLASS Final\whole_model\best_model_2_e50_lr=0.0001.pth")
    model.to(device)

    # Steo 4：网络推理
    model.eval()
    i = 0  # 初始化图片数量
    num_classes = 3  # 分类数

    classes = ('Barlow', "FED", 'Normal')
    # 实例化混淆矩阵
    confusion = ConfusionMatrix(num_classes=num_classes, labels=classes)
    A2C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A3C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A4C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    a2c_num = a3c_num = a4c_num = 0
    start_time = time.time()
    with torch.no_grad():
        for data in test_dataloader:
            if i < len(test_dataloader.dataset.images):
                img_item = test_dataloader.dataset.images[i]
                i += 1
            img, label = data
            img = img.to(device)
            label = label.to(device)
            view_type = img_item.split(os.sep)[-2]
            # 开始计时
            prediction_start_time = time.time()
            output = model(img)
            # 结束计时
            prediction_end_time = time.time()
            ret, predicted_label = torch.max(output, 1)

            # 计算预测时间并累积总时间
            prediction_time = prediction_end_time - prediction_start_time
            print_time = 1000 * prediction_time
            # 更新混淆矩阵
            confusion.update(predicted_label, label)
            if view_type == "A2C":
                A2C_confusion.update(predicted_label, label)
                a2c_num += 1
            if view_type == "A3C":
                A3C_confusion.update(predicted_label, label)
                a3c_num += 1
            if view_type == "A4C":
                A4C_confusion.update(predicted_label, label)
                a4c_num += 1

            print('Image Name:{},predict:{},time:{:.1f} ms'.format(img_item, classes[predicted_label.data.item()],
                                                                   print_time))
            print("_________________________________")
    end_time = time.time()
    total_prediction_time = end_time - start_time

    # 输出混淆矩阵结果
    confusion.plot()
    confusion.summary(plot=True)
    print("A2C视图：")
    print("数量：", a2c_num)
    A2C_confusion.plot()
    A2C_confusion.summary(plot=True)
    print("A3C视图：")
    print("数量：", a3c_num)
    A3C_confusion.plot()
    A3C_confusion.summary(plot=True)
    print("A4C视图：")
    print("数量：", a4c_num)
    A4C_confusion.plot()
    A4C_confusion.summary(plot=True)
    print("how many images:", i)
    print("Total prediction time: {:.2f} seconds".format(total_prediction_time))


if __name__ == '__main__':
    main()

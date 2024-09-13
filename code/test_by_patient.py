import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from Dataset import MVP_Dataset
from resnet18 import Resnet18
from PIL import Image
import csv
from collections import Counter
import time
from matrix import ConfusionMatrix

test_path = r"H:\二尖瓣项目\dataset\final 3 class dataset\test"


# 将类型与标签一一对应
def get_name2label(path):
    name2label = {}
    for name in sorted(os.listdir(os.path.join(path))):
        if not os.path.isdir(os.path.join(path, name)):
            continue
        name2label[name] = len(name2label.keys())
    print(name2label)
    return name2label


name2label = get_name2label(test_path)

# type1=Barlow
type1 = os.listdir(test_path)[0]

# type2=FED
type2 = os.listdir(test_path)[1]

# type3=Normal
type3 = os.listdir(test_path)[2]

# 文件夹目录
Barlow_dir = os.path.join(test_path, type1)
FED_dir = os.path.join(test_path, type2)
Normal_dir = os.path.join(test_path, type3)

Barlow_patient_list = os.listdir(Barlow_dir)
FED_patient_list = os.listdir(FED_dir)
Normal_patient_list = os.listdir(Normal_dir)

""" 
disease_dir: E:\\二尖瓣项目\\MVP\\8_framedata\\xxx
patient_name_list:['Patient1', 'Patient10', 'Patient100','....']
"""


# 获取患者的路径
def get_patient_path(disease_dir, patient_name_list):
    patient_list = []
    for patient_name in patient_name_list:
        patient_dir = os.path.join(disease_dir, patient_name)
        if os.path.exists(patient_dir):
            patient_list.append(patient_dir)
    return patient_list


Barlow_patient_path_list = get_patient_path(Barlow_dir, Barlow_patient_list)
Normal_patient_path_list = get_patient_path(Normal_dir, Normal_patient_list)
FED_patient_path_list = get_patient_path(FED_dir, FED_patient_list)

"""
pathient_path_list:['E:\\二尖瓣项目\\MVP\\8_framedata\\Barlow\\Patient1', 'E:\\二尖瓣项目\\MVP\\8_framedata\\Barlow\\Patient10',...]
"""


# 按照病人的视图路径
def get_view_type(pathient_path_list):
    patients_view_path_list = []
    view_type = ["A2C", "A3C", "A4C", "PLAX"]
    for path in pathient_path_list:
        for view in view_type:
            view_path = os.path.join(path, view)
            if os.path.exists(view_path):
                patients_view_path_list.append(view_path)
    return patients_view_path_list


barlow_patient_view_path_list = get_view_type(Barlow_patient_path_list)
normal_patient_view_path_list = get_view_type(Normal_patient_path_list)

"""
patient_view_path:['E:\\二尖瓣项目\\MVP\\8_framedata\\Barlow\\Patient1\\A2C', 'E:\\二尖瓣项目\\MVP\\8_framedata\\Barlow\\Patient1\\A3C',...]
"""


# 撰写标签的文档
def write_label_by_view(patient_view_path):
    filename = "test_images.csv"
    for path in patient_view_path:
        images = []
        picture_list = os.listdir(path)

        # item_path = os.path.join(path, filename)
        # if (os.path.exists(item_path)):  # 判断文件夹是否存在
        #     os.remove(item_path)
        # print(item_path+"已删除")
        for name in picture_list:
            picture_path = os.path.join(path, name)
            if os.path.exists(picture_path):
                images.append(picture_path)
        with open(os.path.join(path, filename), mode="w", newline="") as f:
            writer = csv.writer(f)
            for img in images:  # E:\\datasets\\pokemon\\bulbasaur\\00000000.png
                name = img.split(os.sep)[-4]
                label = name2label[name]
                # E:\\datasets\\pokemon\\bulbasaur\\00000000.png   ,0
                writer.writerow([img, label])
            print("writen into csv file:", os.path.join(path, filename))


# 删除所有"test_images.csv"文件
def delete_csv_by_view(patient_view_path):
    filename = "test_images.csv"
    for path in patient_view_path:
        images = []
        picture_list = os.listdir(path)
        item_path = os.path.join(path, filename)
        if (os.path.exists(item_path)):  # 判断文件夹是否存在
            os.remove(item_path)
        print(item_path + "已删除")


def test(test_dataloader, model, device):
    i = 0
    T = F = TP = TN = P = N = 0
    correct_sample = 0
    total_sample = 0
    classes = ('Barlow', 'Normal')
    predict = ''
    with torch.no_grad():
        for data in test_dataloader:

            if i < len(test_dataloader.dataset.images):
                img_item = test_dataloader.dataset.images[i]
                i += 1
            img, label = data
            # print(this_img == img)
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            truth = img_item.split("\\")[-4]
            _, predicted_label = torch.max(output, 1)

            if truth == classes[0]:
                T += 1
                TP += (predicted_label == label).cpu().numpy()[0]

            else:
                F += 1
                TN += (predicted_label == label).cpu().numpy()[0]
            P += (predicted_label.data.item() == 0)
            N += (predicted_label.data.item() == 1)
            correct_sample += (predicted_label == label).cpu().numpy()[0]
            total_sample += 1
            print("correct_sample:", correct_sample, "total_sample:", total_sample)
            # # print("correct_sample_1:", correct_sample_1, "total_sample:", total_sample)
            print('Image Name:{},predict:{}'.format(img_item, classes[predicted_label.data.item()]))
            # print("_________________________________")

    return P, N, truth, TP, TN, total_sample


def count_result(T, F, P, TP, TN, correct_sample, total_sample):
    acc = correct_sample / total_sample if total_sample != 0 else 0
    FN = T - TP
    FP = F - TN
    precision = TP / P if P != 0 else 0
    recall = TP / T if T != 0 else 0
    F1_Score = 2 * recall * precision / (recall + precision) if (precision + recall) != 0 else 0
    TPR = TP / T if T != 0 else 0
    TNR = TN / F if F != 0 else 0
    return acc, F1_Score, TPR, TNR


# 定义函数按照患者视图来测试
"""
patient_path_list:['E:\\二尖瓣项目\\MVP\\8_framedata\\Barlow\\Patient1', 'E:\\二尖瓣项目\\MVP\\8_framedata\\Barlow\\Patient10',...]
"""


def test_by_patient(patient_path_list):
    best_models_dict = {"A2C": "new_models_a2c_1.1/best_model_5_e50_lr=0.001.pth",
                        "A3C": "new_models_a3c_1.1/best_model_5_e50_lr=0.0001.pth",
                        "A4C": "new_models_a4c_1.1/best_model_5_e50_lr=0.001.pth",
                        "PLAX": "new_models_plax_1.1/best_model_5_e50_lr=0.001.pth"}
    # best_models_dict = {"A2C": "whole_model/best_model_4_e100_lr=0.0005.pth",
    #                     "A3C": "whole_model/best_model_4_e100_lr=0.0005.pth",
    #                     "A4C": "whole_model/best_model_4_e100_lr=0.0005.pth",
    #                     "A5C": "whole_model/best_model_4_e100_lr=0.0005.pth",
    #                     "PLAX": "whole_model/best_model_4_e100_lr=0.0005.pth"}
    classes = ('Barlow', 'FED', 'Normal')
    wrong_list = []
    patient_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    # 初始化各种统计变量
    total_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A2C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A3C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A4C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    predict_result_list = []
    view_type_list = ["A2C", "A3C", "A4C"]
    predict2tensor = {'Barlow': np.array([0]), 'FED': np.array([1]), 'Normal': np.array([2])}
    start_time = time.time()
    # 遍历每个患者的路径
    for patient_path in patient_path_list:
        patients_view_path_list = []
        truth = patient_path.split("\\")[-2]
        patient_name = truth + "_" + patient_path.split("\\")[-1]
        patient_A2C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
        patient_A3C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
        patient_A4C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
        # 遍历每个视图类型
        for view in view_type_list:
            # if view != "PLAX":
            view_path = os.path.join(patient_path, view)
            if os.path.exists(view_path):
                patients_view_path_list.append(view_path)
                # if patient_name not in patient_name_list:
                #     patient_name_list.append(patient_name)
        # 如果患者有可用的心脏视图
        if len(patients_view_path_list) != 0:

            patient_time = 0
            # 遍历每个视图路径
            for path in patients_view_path_list:
                # for path in patient_view_path:
                view_time = 0
                view_type = path.split("\\")[-1]

                # step1:装载数据集
                dataset_test = MVP_Dataset(path, 224, mode="test")
                test_dataloader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

                # step2 初始化网络
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Step 3：加载训练好的权重
                model = torch.load(best_models_dict[view_type])
                model.to(device)

                # Steo 4：网络推理
                model.eval()

                i = 0
                total_time = 0
                this_view_confusion = ConfusionMatrix(num_classes=3, labels=classes)

                # step 5:结果预测
                with torch.no_grad():
                    for data in test_dataloader:

                        if i < len(test_dataloader.dataset.images):
                            img_item = test_dataloader.dataset.images[i]
                            i += 1

                        img, label = data
                        # print(this_img == img)
                        img = img.to(device)
                        label = label.to(device)
                        # 开始计时
                        prediction_start_time = time.time()
                        output = model(img)
                        # 结束计时
                        prediction_end_time = time.time()
                        _, predicted_label = torch.max(output, 1)
                        # 计算预测时间并累积总时间
                        prediction_time = prediction_end_time - prediction_start_time
                        total_time += prediction_time
                        total_confusion.update(predicted_label, label)
                        this_view_confusion.update(predicted_label, label)
                        if view_type == "A2C":
                            A2C_confusion.update(predicted_label, label)
                            patient_A2C_confusion.update(predicted_label, label)
                        if view_type == "A3C":
                            A3C_confusion.update(predicted_label, label)
                            patient_A3C_confusion.update(predicted_label, label)
                        if view_type == "A4C":
                            A4C_confusion.update(predicted_label, label)
                            patient_A4C_confusion.update(predicted_label, label)

                    # 输出混淆矩阵结果
                this_view_confusion.plot()
                this_view_acc, this_view_predict = this_view_confusion.summary()

                # 打印预测结果和相关统计信息
                print("{} 的 {} 视图的准确率为{}".format(patient_name, view_type, this_view_acc))
                print("{} 的 {} 视图的预测结果为{}".format(patient_name, view_type, this_view_predict))
                view_time += total_time
                patient_time += view_time
                print_time = 1000 * view_time
                print("预测时间为：{:.1f} ms".format(print_time))
                print("_______________________________________________________")
                predict_result_list.append(this_view_predict)

        # 统计预测结果并得出最终结果
        number = Counter(predict_result_list)
        result = number.most_common()

        if len(result) == 1:
            final_result = result[0][0]
        else:
            count_list = []
            Barlow_num = sum(patient_A2C_confusion.matrix[0] + patient_A3C_confusion.matrix[0] + \
                             patient_A4C_confusion.matrix[0])
            FED_num = sum(patient_A2C_confusion.matrix[1] + patient_A3C_confusion.matrix[1] + \
                          patient_A4C_confusion.matrix[1])
            Normal_num = sum(patient_A2C_confusion.matrix[2] + patient_A3C_confusion.matrix[2] + \
                             patient_A4C_confusion.matrix[2])
            count_list.append(Barlow_num)
            count_list.append(FED_num)
            count_list.append(Normal_num)
            max_index = count_list.index(max(count_list))
            final_result = classes[max_index]
        patient_result = predict2tensor[final_result]
        patient_truth = predict2tensor[truth]
        patient_confusion.update(patient_result, patient_truth)
        if final_result != truth:
            wrong_list.append(patient_name)
        print("\n{} 的预测结果为：{} ,真实结果为：{} ".format(patient_name, final_result, truth))
        print("{} 预测时间为：{:.1f} ms".format(patient_name, 1000 * patient_time))
        print("****************************************")


    #  step 6:打印输出结果
    total_sample = len(patient_path_list)
    end_time = time.time()
    total_prediction_time = end_time - start_time
    average_fps = total_sample / total_prediction_time

    print("Patient ACC:")
    patient_confusion.plot()
    patient_confusion.summary(plot=True)
    print("A2C视图：")
    A2C_confusion.summary(plot=True)
    print("A3C视图：")
    A3C_confusion.summary(plot=True)
    print("A4C视图：")
    A4C_confusion.summary(plot=True)
    print("所有视图：")
    total_confusion.summary(plot=True)
    print("Total prediction time: {:.2f} seconds".format(total_prediction_time))
    print("Average FPS: {:.2f}".format(average_fps))
    print("wrong patients:", wrong_list)


if __name__ == "__main__":
    # test_by_patient(Barlow_patient_path_list)
    # print(os.listdir(r"E:\二尖瓣项目\MVP\8_framedata\Barlow"))
    # print(len(Barlow_patient_path_list), len(Normal_patient_path_list),
    #       len(Barlow_patient_path_list + Normal_patient_path_list))
    # print(Barlow_patient_path_list + Normal_patient_path_list)

    barlow_patient_view_path_list = get_view_type(Barlow_patient_path_list)
    normal_patient_view_path_list = get_view_type(Normal_patient_path_list)
    FED_patient_view_path_list = get_view_type(FED_patient_path_list)
    # delete_csv_by_view(barlow_patient_view_path_list)
    # delete_csv_by_view(normal_patient_view_path_list)
    # delete_csv_by_view(FED_patient_view_path_list)
    # write_label_by_view(barlow_patient_view_path_list)
    # write_label_by_view(normal_patient_view_path_list)
    # write_label_by_view(FED_patient_view_path_list)
    all_patient_path_list = Barlow_patient_path_list + FED_patient_path_list + Normal_patient_path_list
    # print(all_patient_path_list)
    test_by_patient(all_patient_path_list)

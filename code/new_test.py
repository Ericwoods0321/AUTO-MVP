import torch
from torch.utils.data import DataLoader
from whole_dataset import MVP_whole_Dataset
from matrix import ConfusionMatrix
import numpy as np
import time


def main():
    # Step 0:查看torch版本、设置device`
    # print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('Barlow', "FED", 'Normal')
    patient_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    # 初始化各种统计变量
    total_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A2C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A3C_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    A4C_confusion = ConfusionMatrix(num_classes=3, labels=classes)

    predict2tensor = {'Barlow': np.array([0]), 'FED': np.array([1]), 'Normal': np.array([2])}
    test_path = r"H:\二尖瓣项目\dataset\final 3 class dataset\test"
    dataset_test = MVP_whole_Dataset(test_path, 224, mode="test")

    test_dataloader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    # Step 3：加载训练好的权重
    model = torch.load(r"H:\二尖瓣项目\3 CLASS Final\whole_model\best_model_2_e50_lr=0.0001.pth")
    model.to(device)

    # Steo 4：网络推理

    correct_sample = 0
    total_sample = 0

    i = 0
    this_patient_count = 0
    patient_list = []
    wrong_list = []
    model.eval()
    total_time = 0
    start_time = time.time()
    this_patient_confusion = ConfusionMatrix(num_classes=3, labels=classes)
    with torch.no_grad():
        for data in test_dataloader:

            if i < len(test_dataloader.dataset.images):
                img_item = test_dataloader.dataset.images[i]
                i += 1
            # print("i", i)
            view_type = img_item.split("\\")[-2]
            # if view_type != "PLAX":
            name = img_item.split("\\")[-3]
            patient_name = img_item.split("\\")[-4] + "_" + name
            if len(patient_list) == 0:
                patient_list.append(patient_name)
            if patient_name not in patient_list:
                this_patient_confusion.plot()
                last_patient = patient_list[-1]
                last_patient_acc, last_patient_predict = this_patient_confusion.summary()
                gt = last_patient.split("_")[0]
                patient_result = predict2tensor[last_patient_predict]
                patient_truth = predict2tensor[gt]
                patient_confusion.update(patient_result, patient_truth)
                print('****************************************************************')
                print("{}预测结果为：{}，真实结果为：{}".format(last_patient, last_patient_predict, gt))
                if last_patient_predict != gt:
                    wrong_list.append([[patient_name], ["predict:", last_patient_predict, "Truth:", gt]])
                print('****************************************************************')
                patient_list.append(patient_name)
                this_patient_confusion.clear()

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

            total_confusion.update(predicted_label, label)
            this_patient_confusion.update(predicted_label, label)
            if view_type == "A2C":
                A2C_confusion.update(predicted_label, label)
            if view_type == "A3C":
                A3C_confusion.update(predicted_label, label)
            if view_type == "A4C":
                A4C_confusion.update(predicted_label, label)

            this_patient_count += 1
            # 计算预测时间并累积总时间
            prediction_time = prediction_end_time - prediction_start_time
            print_time = 1000 * prediction_time
            total_time += prediction_time
            correct_sample += (predicted_label == label).cpu().numpy()[0]

            total_sample += 1
            print("correct_sample:", correct_sample, "total_sample:", total_sample)

            print('Image Name:{},predict:{},time:{:.1f} ms'.format(img_item, classes[predicted_label.data.item()],
                                                                   print_time))

            print("_________________________________")
            if i == len(test_dataloader.dataset.images):
                this_patient_confusion.plot()
                last_patient = patient_name
                last_patient_acc, last_patient_predict = this_patient_confusion.summary()
                gt = last_patient.split("_")[0]
                patient_result = predict2tensor[last_patient_predict]
                patient_truth = predict2tensor[gt]
                patient_confusion.update(patient_result, patient_truth)
                print('****************************************************************')
                print("{}预测结果为：{}，真实结果为：{}".format(last_patient, last_patient_predict, gt))
                if last_patient_predict != gt:
                    wrong_list.append([[patient_name], ["predict:", last_patient_predict, "Truth:", gt]])
                print('****************************************************************')
                patient_list.append(patient_name)
                this_patient_confusion.clear()

    end_time = time.time()
    total_prediction_time = end_time - start_time
    average_fps = total_sample / total_prediction_time
    patient_fps = (len(patient_list) / total_prediction_time)

    print("\n")
    # Step 5:打印分类准确率
    print("Total prediction time: {:.2f} seconds".format(total_prediction_time))
    print("Patient FPS: {:.2f}".format(patient_fps))

    print("Average FPS: {:.2f}".format(average_fps))
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
    print("wrong_list:", wrong_list)


if __name__ == '__main__':
    main()

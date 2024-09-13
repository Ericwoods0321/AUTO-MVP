import torch
from torch.utils.data import DataLoader
from Dataset import MVP_Dataset
from whole_dataset import MVP_whole_Dataset
import time


def main():
    # Step 0:查看torch版本、设置device
    # print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('Barlow', 'Normal')

    test_path = r"dataset/mvp_dataset/A3C/val"
    dataset_test = MVP_whole_Dataset(test_path, 224, mode="test")
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    # Step 3：加载训练好的权重
    model = torch.load("module/A3C/A3C_best_model_2_e100_lr=0.0001.pth")
    model.to(device)

    # Steo 4：网络推理
    model.eval()

    correct_sample = 0
    total_sample = 0
    T = F = TP = TN = P = N = 0
    i = 0
    total_time = 0
    start_time = time.time()
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
            print_time = 1000 * prediction_time
            total_time += prediction_time
            if img_item.split("\\")[-2] == classes[0]:
                T += 1
                TP += (predicted_label == label).cpu().numpy()[0]

            else:
                F += 1
                TN += (predicted_label == label).cpu().numpy()[0]
            correct_sample += (predicted_label == label).cpu().numpy()[0]
            P += (predicted_label.data.item() == 0)
            N += (predicted_label.data.item() == 1)
            # correct_sample_1 += (img_item.split("\\")[-2] == classes[predicted_label.data.item()])
            total_sample += 1
            print("correct_sample:", correct_sample, "total_sample:", total_sample)
            # print("correct_sample_1:", correct_sample_1, "total_sample:", total_sample)
            print('Image Name:{},predict:{},time:{:.1f} ms'.format(img_item, classes[predicted_label.data.item()],
                                                                   print_time))
            print("_________________________________")
    end_time = time.time()
    total_prediction_time = end_time - start_time
    average_fps = total_sample / total_prediction_time
    FN = T - TP
    FP = F - TN
    precision = TP / P
    recall = TP / T
    F1_Score = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0
    # Step 5:打印分类准确率
    print("acc:", correct_sample / total_sample)
    print("正样本数量T：", T, "负样本数量F：", F, "样本总数：", total_sample)
    print("TP数量：", TP, "FP数量：", FP, "P的数量：", P, "\nFN数量：", FN, "TN数量：", TN, "N的数量：", N)
    print("precision:", precision, "recall:", recall)
    print("F1-Score:", F1_Score)
    if T != 0 and F != 0:
        print("True positive rate(TPR)：", TP / T, "\nTrue negative rate(TNR)：", TN / F)
    if T != 0 and F == 0:
        print("True positive rate(TPR)：", TP / T, "F为0，无法计算TNR,TN={},F={}".format(TN, F))
    if T == 0 and F != 0:
        print("True positive rate(TnR)：", TN / F, "T为0，无法计算TPR,TP={},T={}".format(TP, T))
    if T == F == 0:
        print("T与F均为0,T={},F={}".format(T, F))

    print("Total prediction time: {:.2f} seconds".format(total_prediction_time))
    print("Average FPS: {:.2f}".format(average_fps))


if __name__ == '__main__':
    main()

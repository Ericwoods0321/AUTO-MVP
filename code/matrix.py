import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self, plot=False):  # 计算指标函数
        # calculate accuracy
        rowsum_list = []
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "f1_score",
                             "Specificity(True negative rate)"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            f1_score = round(2 * Precision * Recall / (Precision + Recall), 3) if (Precision + Recall) != 0 else 0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, f1_score, Specificity])

            row_sum = sum(self.matrix[i])
            rowsum_list.append(row_sum)
        if plot:
            print(table)
        max_index = rowsum_list.index(max(rowsum_list))
        predict = self.labels[max_index]
        return str(acc), predict

    def plot(self, plot=False):  # 绘制混淆矩阵
        matrix = self.matrix
        np.set_printoptions(suppress=True)
        print(matrix)
        if plot:
            plt.imshow(matrix, cmap=plt.cm.Blues)

            # 设置x轴坐标label
            plt.xticks(range(self.num_classes), self.labels, rotation=45)
            # 设置y轴坐标label
            plt.yticks(range(self.num_classes), self.labels)
            # 显示colorbar
            plt.colorbar()
            plt.xlabel('True Labels')
            plt.ylabel('Predicted Labels')
            plt.title('Confusion matrix (acc=' + self.summary()[0] + ')')

            # 在图中标注数量/概率信息
            thresh = matrix.max() / 2
            for x in range(self.num_classes):
                for y in range(self.num_classes):
                    # 注意这里的matrix[y, x]不是matrix[x, y]
                    info = int(matrix[y, x])
                    plt.text(x, y, info,
                             verticalalignment='center',
                             horizontalalignment='center',
                             color="white" if info > thresh else "black")
            plt.tight_layout()
            plt.show()

    def clear(self):
        zeros_arr = np.zeros_like(self.matrix)
        self.matrix = zeros_arr

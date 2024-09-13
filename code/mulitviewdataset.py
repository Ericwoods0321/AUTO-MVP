import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from mulit import MultiBranch
from torch.autograd import Variable


class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.views = ['A2C', 'A3C', 'A4C', 'A5C', 'PLAX']
        self.num_views = len(self.views)
        self.num_images_per_view = 8
        self.samples = []
        for label, class_name in enumerate(['Normal','Barlow' ]):
            class_dir = os.path.join(root_dir, class_name)
            for case_name in os.listdir(class_dir):
                case_dir = os.path.join(class_dir, case_name)
                sample = {'images': [], 'label': label}
                for view in self.views:
                    view_dir = os.path.join(case_dir, view)
                    images = sorted(os.listdir(view_dir))[:self.num_images_per_view]
                    sample['images'].append([os.path.join(view_dir, img) for img in images])
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = sample['images']
        label = sample['label']
        data = []
        for view in range(self.num_views):
            view_data = []
            for img_path in images[view]:
                img = Image.open(img_path).convert('L')
                if self.transform:
                    img = self.transform(img)
                view_data.append(img)
            data.append(torch.stack(view_data))
        data = torch.stack(data)
        return data, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    import torch.optim as optim
    # 创建一个变换对象，用于对图像进行预处理
    transform = Compose([
        Resize((128, 128)),  # 将图像的大小统一为 128 x 128
        ToTensor(),  # 将图像转换为张量，并归一化到 [0, 1] 范围内
        Normalize([0.5], [0.5])
    ])

    # 创建一个数据集对象，指定数据集的根目录和变换对象
    dataset = MultiViewDataset('E:\\二尖瓣项目\\code\\Dataset\\mulit_dataset\\train', transform=transform)

    # 创建一个数据加载器对象，指定数据集对象、批量大小和是否打乱数据
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    model = MultiBranch()

    device = torch.device('cpu')
    model.to(device)
    correct = 0
    optimizer = optim.Adam(model.parameters(), lr=1*10-3)
    # 在训练循环中，使用数据加载器对象来获取批量数据
    for batch_idx, (data, label) in enumerate(data_loader):
        # 使用data和label进行训练
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        preds = (out > 0.5).long().squeeze()
        correct += torch.sum(preds == label)
        labels = label.float().view(-1, 1)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(batch_idx, loss)

import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
datapath = "pokemon"


class MVP_Dataset(Dataset):

    def __init__(self, root, resize, mode):
        super(MVP_Dataset, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        self.mode = mode
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        self.images, self.labels = self.load_csv(mode + "_" + "images.csv")

    def load_csv(self, filename):

        if os.path.exists(os.path.join(self.root, filename)) == 0:
            images = []
            for name in self.name2label.keys():
                disease_path = os.path.join(self.root, name)

                for items in os.listdir(disease_path):
                    image_path = os.path.join(disease_path, items)
                    images.append(image_path)

            # print(len(images),images)
            # {bulbasaur:0,charmander:1,mewtwo:2   }
            # random.shuffle(images)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:  # E:\\datasets\\pokemon\\bulbasaur\\00000000.png
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # E:\\datasets\\pokemon\\bulbasaur\\00000000.png   ,0
                    writer.writerow([img, label])
                print("writen into csv file:", filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):

        # idx [0-len(images)]
        # self.images,self.labels
        # img:"pokemon\\bulbasaur\\0000000.png"   label :0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert("RGB"),  # string path=>image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # mean,std为统计常量，给图像归一化
        ])
        img = tf(img)
        label = torch.tensor(label)

        return img, label


if __name__ == "__main__":
    train_path = r"E:\二尖瓣项目\code\Dataset\dataset\A5C\train"
    val_path = r"E:\二尖瓣项目\code\Dataset\dataset\A5C\val"
    test_path = r"E:\二尖瓣项目\code\Dataset\dataset\A5C\test"
    a = MVP_Dataset(train_path, 224, "train")
    a = MVP_Dataset(val_path, 224, "val")
    a = MVP_Dataset(test_path, 224, "test")

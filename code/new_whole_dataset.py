import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
datapath = "pokemon"


class MVP_whole_Dataset_new(Dataset):

    def __init__(self, root, resize, mode):
        super(MVP_whole_Dataset_new, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        self.mode = mode
        self.view_list = ["A2C", "A3C", "A4C", "A5C", "PLAX"]
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        self.images, self.views, self.labels = self.load_csv("new_" + mode + "_" + "images.csv")

    def load_csv(self, filename):

        if os.path.exists(os.path.join(self.root, filename)) == 0:
            images, views = [], []
            for name in self.name2label.keys():
                disease_path = os.path.join(self.root, name)
                patient_list = os.listdir(disease_path)

                for patients in patient_list:
                    patient_path = os.path.join(disease_path, patients)
                    patient_view_list = os.listdir(patient_path)
                    for view in patient_view_list:
                        if view in self.view_list:
                            patient_view_path = os.path.join(patient_path, view)
                            img_list = os.listdir(patient_view_path)
                            for img in img_list:
                                img_path = os.path.join(patient_view_path, img)
                                images.append(img_path)
                                views.append(view)

            # print(len(images),images)
            # {bulbasaur:0,charmander:1,mewtwo:2   }
            # random.shuffle(images)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for i in range(len(images)):  # E:\\datasets\\pokemon\\bulbasaur\\00000000.png
                    img = images[i]
                    name = img.split(os.sep)[-4]
                    label = self.name2label[name]
                    # E:\\datasets\\pokemon\\bulbasaur\\00000000.png   ,0
                    view = views[i]
                    writer.writerow([img, view, label])
                print("writen into csv file:", filename)

        # read from csv file
        images, views, labels = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, view, label = row
                label = int(label)
                images.append(img)
                views.append(view)
                labels.append(label)

        assert len(images) == len(labels)

        return images, views, labels

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
        img, view, label = self.images[idx], self.views[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert("RGB"),  # string path=>image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # mean,std为统计常量，给图像归一化
        ])

        view_index = self.view_list.index(view)
        img = tf(img)
        view_type = torch.eye(len(self.view_list))[view_index]
        label = torch.tensor(label)

        return img, view_type, label


if __name__ == "__main__":
    train_path = r"H:\二尖瓣项目\dataset\whole_data\train"

    a = MVP_whole_Dataset_new(train_path, 224, "train")
    print(a)

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_path, label_path="", train=True):
        super(Dataset, self).__init__()
        self.img_path = image_path
        self.image_names = os.listdir(image_path)
        self.len = len(self.image_names)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train = train
        if self.train:
            self.label_path = label_path

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        name = self.image_names[index]
        imagepath = os.path.join(self.img_path, name)
        img = self.transform(Image.open(imagepath))
        if self.train:
            labelpath = os.path.join(self.label_path, name)
            label = self.transform(Image.open(labelpath))
            return img, label
        else:
            return img, name


if __name__ == '__main__':
    dataset = MyDataset('', '')
    len(dataset)


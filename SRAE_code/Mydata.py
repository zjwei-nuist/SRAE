from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def channel_F_to_L(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img

def channel_L_to_F(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(1, 2)
    return img


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader('D:' + fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
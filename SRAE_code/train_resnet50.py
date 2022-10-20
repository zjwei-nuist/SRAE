import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
from torch import nn
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

root = 'D:/readv/caltech256/'


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
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std),
])

if __name__ == "__main__":
    use_cuda = True
    image_nc = 3
    batch_size = 128

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    train_data = MyDataset(txt=root + 'dataset-trn.txt', transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    test_data = MyDataset(txt=root + 'dataset-val.txt', transform=transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, pin_memory=True, num_workers=4)

    # # training the target model
    target_model = models.resnet50(pretrained=True)
    for param in target_model.parameters():
        param.requires_grad = False

    fc_features = target_model.fc.in_features
    target_model.fc = nn.Linear(fc_features, 257)

    target_model.to(device)
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.01, betas=(0.9, 0.999))

    epochs = 30
    best_accuracy = 0
    for epoch in range(epochs):

        if epoch == 10:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        loss_epoch = 0
        target_model.train()
        for i, data in enumerate(train_loader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()  # 将导数置0
            loss_model.backward()
            opt_model.step()
        print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))

        num_correct = 0
        target_model.eval()
        for i, data in enumerate(test_loader, 0):
            test_img, test_label = data
            test_img, test_label = test_img.to(device), test_label.to(device)
            pred_lab = torch.argmax(target_model(test_img), 1)
            num_correct += torch.sum(pred_lab == test_label, 0)

        acc = (num_correct.item() / len(test_data))
        if acc > best_accuracy:
            best_accuracy = acc
            targeted_model_file_name = './ResNet50.pth'
            torch.save(target_model.state_dict(), targeted_model_file_name)
            print("epoch %d, best accuracy %g" % (epoch, best_accuracy))

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from RGAN import Attack
from torchvision import models
from Mydata import MyDataset
from torch import nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = True
image_nc = 3
epochs = 150
BOX_MIN = 0  # 图像归一化
BOX_MAX = 1
root = './'

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Define what device we are using
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
model_num_labels = 257

batch_size = 22
pretrained_model = "./DenseNet121.pth"
targeted_model = models.densenet121(pretrained=False)
fc_features = targeted_model.classifier.in_features
targeted_model.classifier = nn.Linear(fc_features, 257)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.to(device)
targeted_model.eval()

# batch_size = 32
# pretrained_model = './ResNet50.pth'
# targeted_model = models.resnet50(pretrained=False)
# for param in targeted_model.parameters():
#     param.requires_grad = False
# fc_features = targeted_model.fc.in_features
# targeted_model.fc = nn.Linear(fc_features, 257)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)
# targeted_model.eval()

# dataset can be download in https://authors.library.caltech.edu/7694/
train_data = MyDataset(txt=root + 'dataset-trn.txt', transform=transform)
dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)


def main():
    clip = 1
    RGAN = Attack(device,
                           targeted_model,
                           model_num_labels,
                           image_nc,
                           BOX_MIN,
                           BOX_MAX,
                           clip)
    RGAN.train(dataloader, epochs)


if __name__ == '__main__':
    main()

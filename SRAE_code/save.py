import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import  models
from torchvision import models as model
import linecache
from Mydata import MyDataset
from torch import nn
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
clip = 1
model_path = './'
root = './'
# train_path = model_path + 'train1/'
test_path = model_path + 'save_test/'

# train_compara = train_path + 'compara/'
# train_img = train_path + 'img/'
# train_adv = train_path + 'adv/'
# train_r = train_path + 'r_adv/'
# train_perturb = train_path + 'pert/'
# train_r_pert = train_path + 'r_pert'
# train_diff = train_path + 'diff/'

# test_compara = test_path + 'compara/'
test_img = test_path + 'img/'
test_adv = test_path + 'adv/'
test_r = test_path + 'r_adv/'
test_perturb = test_path + 'pert1/'
test_perturb5 = test_path + 'pert5/'
test_r_pert = test_path + 'r_pert1/'
test_r_pert5 = test_path + 'r_pert5/'
test_diff = test_path + 'diff1/'
test_diff5 = test_path + 'diff5/'

# all_path = [train_path, test_path, train_img, train_adv, train_r,
#             train_perturb, train_r_pert, train_diff, test_img, test_adv,
#             test_r, test_perturb, test_perturb5,test_r_pert5, test_r_pert, test_diff, train_compara, test_compara]

all_path = [test_img, test_adv, test_r, test_perturb, test_perturb5, test_r_pert5,
            test_r_pert, test_diff, test_diff5]

for i in range(len(all_path)):
    if not os.path.exists(all_path[i]):
        os.makedirs(all_path[i])


def draw_scatter(x_value, y_value, x_label, y_label, save_path):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x_value, y_value, s=1)
    plt.savefig(save_path)
    plt.close()


def channel_first_to_last(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img


def show(test_img, test_label, path):
    label = []
    test_img = test_img.detach().cpu()
    test_label = test_label.detach().cpu()
    for i, data in enumerate(test_label):
        label.append(linecache.getline('./label.txt', data.item() + 1))
    for j in range(len(test_img)):
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(label[j].split('.')[1])
        plt.imshow(channel_first_to_last(test_img[j]))
        plt.savefig(path + ".png")


use_cuda = True
image_nc = 3
batch_size = 30

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

gen_input_nc = image_nc

# Define what device we are using
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

test_data = MyDataset(txt=root + 'dataset-val.txt', transform=transform)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, pin_memory=True, num_workers=1)

# load the generator of adversarial examples
pretrained_generator_path = model_path + 'netG_epoch_150_1.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# load the remover of adversarial examples
pretrained_remover_path = model_path + 'netR_epoch_150_1.pth'
pretrained_R = models.Remover(gen_input_nc, image_nc).to(device)
pretrained_R.load_state_dict(torch.load(pretrained_remover_path))
pretrained_R.eval()

pretrained_model = "./DenseNet121.pth"

target_model = model.densenet121(pretrained=False)
fc_features = target_model.classifier.in_features
target_model.classifier = nn.Linear(fc_features, 257)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.to(device)
target_model.eval()


def main():
    count = 0
    num = 0
    num_correct = 0
    num_correct_r = 0

    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -clip, clip)
        adv_img = perturbation + test_img

        adv_img = torch.clamp(adv_img, 0, 1)
        # adv_img = Quantization(adv_img)

        pred_lab = torch.argmax(target_model(adv_img), 1)
        num_correct += torch.sum(pred_lab == test_label, 0)

        r_perturbation = pretrained_R(adv_img)
        r_adv = adv_img - r_perturbation

        r_adv = torch.clamp(r_adv, 0, 1)
        # r_adv = Quantization(r_adv)

        pred_r_adv = torch.argmax(target_model(r_adv), 1)
        num_correct_r += torch.sum(pred_r_adv == test_label, 0)

        ori_pred = torch.argmax(target_model(test_img), 1)
        num += torch.sum(ori_pred == test_label, 0)

        for j in range(len(test_img)):
            # use uint8
            # r_a = (r_adv[j] * 255.).detach().cpu().numpy().astype('uint8').squeeze()
            # ori = (test_img[j] * 255.).detach().cpu().numpy().astype('uint8').squeeze()
            # adv = (adv_img[j] * 255.).detach().cpu().numpy().astype('uint8').squeeze()

            # r_a = channel_first_to_last(r_a)
            # ori = channel_first_to_last(ori)
            # adv = channel_first_to_last(adv)

            p = adv_img[j] - test_img[j]
            p = torch.abs(p)
            r_p = r_adv[j] - adv_img[j]
            r_p = torch.abs(r_p)
            diff = r_p - p
            diff = torch.abs(diff)

            img2 = transforms.ToPILImage()(torch.squeeze(adv_img[j]).cpu()).convert('RGB')
            ori = transforms.ToPILImage()(torch.squeeze(test_img[j]).cpu()).convert('RGB')
            r_a = transforms.ToPILImage()(torch.squeeze(r_adv[j]).cpu()).convert('RGB')

            p1 = transforms.ToPILImage()(torch.squeeze(p).cpu()).convert('RGB')
            p5 = transforms.ToPILImage()(torch.squeeze(p * 10).cpu()).convert('RGB')

            r_p1 = transforms.ToPILImage()(torch.squeeze(r_p).cpu()).convert('RGB')
            r_p5 = transforms.ToPILImage()(torch.squeeze(r_p * 10).cpu()).convert('RGB')

            diff1 = transforms.ToPILImage()(torch.squeeze(diff).cpu()).convert('RGB')
            diff5 = transforms.ToPILImage()(torch.squeeze(diff * 10).cpu()).convert('RGB')

            ori.save(test_path + 'img/' + str(count) + '_ori_.png')
            img2.save(test_path + 'adv/' + str(count) + '_' + str(pred_lab[j].item())
                      + '_' + str(test_label[j].item()) + '_.png')
            r_a.save(test_path + 'r_adv/' + str(count) + '_' + str(pred_lab[j].item())
                     + '_' + str(test_label[j].item()) + '_' + str(pred_r_adv[j].item()) + '_r_a_.png')

            p1.save(test_path + 'pert1/' + str(count) + '_noise1_.png')
            p5.save(test_path + 'pert5/' + str(count) + '_noise5_.png')

            r_p1.save(test_path + 'r_pert1/' + str(count) + '_rnoise1_.png')
            r_p5.save(test_path + 'r_pert5/' + str(count) + '_rnoise5_.png')

            diff1.save(test_path + 'diff1/' + str(count) + '_diff1_.png')
            diff5.save(test_path + 'diff5/' + str(count) + '_diff5_.png')

            count += 1


if __name__ == '__main__':
    main()
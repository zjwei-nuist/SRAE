import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from torchvision import models as model
from Mydata import MyDataset
from torch import nn

import numpy as np
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
clip = 1
model_path = './'
root = 'D:/readv/caltech256/'


def channel_first_to_last(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img


use_cuda = True
image_nc = 3

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

gen_input_nc = image_nc

# Define what device we are using
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

# load the generator of adversarial examples
pretrained_generator_path = model_path + 'netG_epoch_150_1.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# load the remover of adversarial examples
pretrained_remover_path = model_path + 'netR_epoch_150_1.pth'
pretrained_R = models.Recover(gen_input_nc, image_nc).to(device)
pretrained_R.load_state_dict(torch.load(pretrained_remover_path))
pretrained_R.eval()

batch_size = 32
pretrained_model = "./DenseNet121.pth"
target_model = model.densenet121(pretrained=False)
fc_features = target_model.classifier.in_features
target_model.classifier = nn.Linear(fc_features, 257)
target_model.load_state_dict(torch.load(pretrained_model))

# batch_size = 32
# pretrained_model = 'D:/models/Caltech256/resnet50/ResNet50.pth'
# target_model = model.resnet50(pretrained=False)
# for param in target_model.parameters():
#     param.requires_grad = False
# fc_features = target_model.fc.in_features
# target_model.fc = nn.Linear(fc_features, 257)
# target_model.load_state_dict(torch.load(pretrained_model))

# batch_size = 56
# pretrained_model = "D:/models/Caltech256/moblieV3/MoblieNetV3.pth"
# target_model = model.mobilenet_v3_large(pretrained=False)
# for param in target_model.parameters():
#     param.requires_grad = False
# fc_features = target_model.classifier[3].in_features
# target_model.classifier[3] = nn.Linear(fc_features, 257)
# target_model.load_state_dict(torch.load(pretrained_model))

target_model.to(device)
target_model.eval()

test_data = MyDataset(txt=root + 'dataset-val.txt', transform=transform)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, pin_memory=True, num_workers=1)

def main():
    count = 0
    num = 0
    num_correct = 0
    num_correct_r = 0

    l0_adv_ori_list = []
    l2_adv_ori_list = []
    l_inf_adv_ori_list = []
    psnr_adv_ori_list = []
    ssim_adv_ori_list = []
    perceptual_adv_ori_list = []

    l0_r_ori_list = []
    l2_r_ori_list = []
    l_inf_r_ori_list = []
    psnr_r_ori_list = []
    ssim_r_ori_list = []
    perceptual_r_ori_list = []

    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -clip, clip)
        adv_img = perturbation + test_img

        adv_img = torch.clamp(adv_img, 0, 1)

        pred_lab = torch.argmax(target_model(adv_img), 1)
        num_correct += torch.sum(pred_lab == test_label, 0)

        r_perturbation = pretrained_R(adv_img)
        r_adv = adv_img - r_perturbation

        r_adv = torch.clamp(r_adv, 0, 1)

        pred_r_adv = torch.argmax(target_model(r_adv), 1)
        num_correct_r += torch.sum(pred_r_adv == test_label, 0)

        ori_pred = torch.argmax(target_model(test_img), 1)
        num += torch.sum(ori_pred == test_label, 0)

        for j in range(len(test_img)):
            # use uint8
            r_a = (r_adv[j]*255.).detach().cpu().numpy().astype('uint8').squeeze()
            ori = (test_img[j]*255.).detach().cpu().numpy().astype('uint8').squeeze()
            adv = (adv_img[j]*255.).detach().cpu().numpy().astype('uint8').squeeze()
            r_a = channel_first_to_last(r_a)
            ori = channel_first_to_last(ori)
            adv = channel_first_to_last(adv)

            l0_adv_ori_list.append(torch.norm((adv_img[j] - test_img[j]), p=0).detach().cpu().numpy().item())
            l2_adv_ori_list.append(torch.norm((adv_img[j] - test_img[j])).detach().cpu().numpy().item())
            l_inf_adv_ori_list.append(torch.norm((adv_img[j] - test_img[j]), p=float('inf')).detach().cpu().numpy().item())
            # perceptual_adv_ori_list.append(loss_fn_vgg(adv_img[j], test_img[j]).item())
            psnr_adv_ori_list.append(compare_psnr(adv, ori, data_range=255))
            ssim_adv_ori_list.append(compare_ssim(adv, ori, multichannel=True))

            l0_r_ori_list.append(torch.norm((r_adv[j] - test_img[j]), p=0).detach().cpu().numpy().item())
            l2_r_ori_list.append(torch.norm((r_adv[j] - test_img[j])).detach().cpu().numpy().item())
            l_inf_r_ori_list.append(torch.norm((r_adv[j] - test_img[j]), p=float('inf')).detach().cpu().numpy().item())
            # perceptual_r_ori_list.append(loss_fn_vgg(r_adv[j], test_img[j]).item())
            psnr_r_ori_list.append(compare_psnr(r_a, ori, data_range=255))
            ssim_r_ori_list.append(compare_ssim(r_a, ori, multichannel=True))

    print('target error rate{:.3f}%'.format(100 * (1 - num.item() / len(test_data))))
    print('generator error rate{:.3f}%'.format(100 * (1 - num_correct.item() / len(test_data))))
    print('remover error rate{:.3f}%'.format(100 * (1 - num_correct_r.item() / len(test_data))))
    print('psnr_adv_ori', np.max(psnr_adv_ori_list), np.min(psnr_adv_ori_list), np.mean(psnr_adv_ori_list),
          np.median(psnr_adv_ori_list), np.var(psnr_adv_ori_list))
    print('ssim_adv_ori', np.max(ssim_adv_ori_list), np.min(ssim_adv_ori_list), np.mean(ssim_adv_ori_list),
          np.median(ssim_adv_ori_list), np.var(ssim_adv_ori_list))
    print('l0_adv_ori', np.max(l0_adv_ori_list), np.min(l0_adv_ori_list), np.mean(l0_adv_ori_list),
          np.median(l0_adv_ori_list), np.var(l0_adv_ori_list))
    print('l2_adv_ori', np.max(l2_adv_ori_list), np.min(l2_adv_ori_list), np.mean(l2_adv_ori_list),
          np.median(l2_adv_ori_list), np.var(l2_adv_ori_list))
    print('l_inf_adv_ori', np.max(l_inf_adv_ori_list), np.min(l_inf_adv_ori_list), np.mean(l_inf_adv_ori_list),
          np.median(l_inf_adv_ori_list), np.var(l_inf_adv_ori_list))
    # print('perceptual_adv_ori', np.max(perceptual_adv_ori_list), np.min(perceptual_adv_ori_list),
    #       np.mean(perceptual_adv_ori_list),
    #       np.median(perceptual_adv_ori_list), np.var(perceptual_adv_ori_list))

    print('psnr_r_ori', np.max(psnr_r_ori_list), np.min(psnr_r_ori_list), np.mean(psnr_r_ori_list),
          np.median(psnr_r_ori_list), np.var(psnr_r_ori_list))
    print('ssim_r_ori', np.max(ssim_r_ori_list), np.min(ssim_r_ori_list), np.mean(ssim_r_ori_list),
          np.median(ssim_r_ori_list), np.var(ssim_r_ori_list))
    print('l0_r_ori', np.max(l0_r_ori_list), np.min(l0_r_ori_list), np.mean(l0_r_ori_list),
          np.median(l0_r_ori_list), np.var(l0_r_ori_list))
    print('l2_r_ori', np.max(l2_r_ori_list), np.min(l2_r_ori_list), np.mean(l2_r_ori_list),
          np.median(l2_r_ori_list), np.var(l2_r_ori_list))
    print('l_inf_r_ori', np.max(l_inf_r_ori_list), np.min(l_inf_r_ori_list), np.mean(l_inf_r_ori_list),
          np.median(l_inf_r_ori_list), np.var(l_inf_r_ori_list))
    # print('perceptual_r_ori', np.max(perceptual_r_ori_list), np.min(perceptual_r_ori_list), np.mean(perceptual_r_ori_list),
    #       np.median(perceptual_r_ori_list), np.var(perceptual_r_ori_list))

    # draw_scatter(psnr_adv_ori_list, perceptual_adv_ori_list, 'PSNR', 'Perceptual',
    #              model_path+'test_adv_ori_psnr_perceptual.png')
    # draw_scatter(psnr_r_ori_list, perceptual_r_ori_list, 'PSNR', 'Perceptual',
    #              model_path+'test_r_ori_psnr_perceptual.png')
    # draw_scatter(l2_adv_ori_list, perceptual_adv_ori_list, 'L2-norm', 'Perceptual',
    #              model_path+'test_adv_ori_l2_perceptual.png')
    # draw_scatter(l2_r_ori_list, perceptual_r_ori_list, 'L2-norm', 'Perceptual',
    #              model_path+'test_r_ori_l2_perceptual.png')
    # draw_scatter(l0_adv_ori_list, l0_r_ori_list,
    #              'L0_adv_ori', 'L0_r_ori', test_path + 'l0_val.png')
    # draw_scatter(l2_adv_ori_list, l2_r_ori_list,
    #              'L2_adv_ori', 'L2_r_ori', test_path + 'l2_val.png')
    # draw_scatter(l_inf_adv_ori_list, l_inf_r_ori_list,
    #              'Linf_adv_ori', 'Linf_r_ori', test_path + 'linf_val.png')

    f = open(model_path + 'test.txt', 'a', encoding='utf-8')
    f.close()


if __name__ == '__main__':
    main()

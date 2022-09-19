import numpy
import torch.nn as nn
import torch
import numpy as np
import models as models
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic


def draw(models_path, data, label):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(data, label=str(label))
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(models_path + str(label) + '1.png')
    plt.close()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 clip):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.clip = clip

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)
        self.netR = models.Recover(self.gen_input_nc, image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)
        self.netR.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)
        self.optimizer_R = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)
        # self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

        self.models_path = './'

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train_batch(self, x, labels):

        for i in range(1):
            perturbation = self.netG(x)

            adv_images = torch.clamp(perturbation, -self.clip, self.clip) + x  

            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()  
            pred_real = self.netDisc(x)  
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))  
            loss_D_real.backward()  

            pred_fake = self.netDisc(adv_images.detach()) 
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device)) 
            loss_D_fake.backward() 
            loss_D_GAN = loss_D_fake + loss_D_real   
            self.optimizer_D.step()  

        # optimize R
        for i in range(1):
            self.optimizer_R.zero_grad()
            r_pert = self.netR(adv_images.detach())
            r_adv = adv_images - r_pert

            r_adv = torch.clamp(r_adv, self.box_min, self.box_max)

            # cal perceptual loss
            # loss_perceptual = self.loss_fn_vgg(r_adv, x)
            # loss_perceptual = torch.mean(loss_perceptual)

            # cal l2 loss
            loss_l2_r_pert = F.mse_loss(r_adv, x)

            # cal adv loss
            logits_model = self.model(r_adv)
            loss_r_adv = F.cross_entropy(logits_model, labels)

            # M1
            # loss_R = loss_r_adv

            # M2
            # adv_lambda = 1
            # pert_lambda = 10
            # loss_R = adv_lambda * loss_r_adv + pert_lambda * loss_l2_r_pert

            # M3
            loss_R = loss_l2_r_pert

            # M4
            # adv_lambda = 1
            # perceptual_lambda = 1
            # loss_R = adv_lambda * loss_r_adv + perceptual_lambda * loss_perceptual

            loss_R.backward(retain_graph=True)
            # loss_R.backward()
            self.optimizer_R.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()  

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)  
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            # cal adv loss
            logits_model = self.model(adv_images) 
            probs_model = F.softmax(logits_model, dim=1)  
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]  

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()  
            self.optimizer_G.step()  

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), \
               loss_adv.item(), loss_r_adv.item(), loss_l2_r_pert.item()

        # return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), \
        #        loss_adv.item(), loss_r_adv.item(), loss_l2_r_pert.item(), loss_perceptual.item()

    def train(self, train_dataloader, epochs):
        loss_D = []
        loss_G_fake = []
        loss_perturb = []
        loss_adv = []
        loss_r_adv = []
        loss_l2_r_pert = []
        # loss_perceptual = []

        for epoch in range(1, epochs+1):
            # time_start = time.time()
            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
                self.optimizer_R = torch.optim.Adam(self.netR.parameters(),
                                                    lr=0.0001)
            if epoch == 100:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
                self.optimizer_R = torch.optim.Adam(self.netR.parameters(),
                                                    lr=0.00001)

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_r_adv_sum = 0
            loss_l2_r_pert_sum = 0
            # loss_perceptual_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_r_adv_batch, \
                loss_l2_r_pert_batch = self.train_batch(images, labels)

                # loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_r_adv_batch, \
                # loss_l2_r_pert_batch, loss_perceptual_batch = self.train_batch(images, labels)

                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_r_adv_sum += loss_r_adv_batch
                loss_l2_r_pert_sum += loss_l2_r_pert_batch
                # loss_perceptual_sum += loss_perceptual_batch

            num_batch = len(train_dataloader)
            loss_D.append(loss_D_sum/num_batch)
            loss_G_fake.append(loss_G_fake_sum/num_batch)
            loss_perturb.append(loss_perturb_sum/num_batch)
            loss_adv.append(loss_adv_sum/num_batch)
            loss_r_adv.append(loss_r_adv_sum/num_batch)
            loss_l2_r_pert.append(loss_l2_r_pert_sum/num_batch)
            # loss_perceptual.append(loss_perceptual_sum/num_batch)

            print("epoch %d:\nloss_D: %.8f, loss_G_fake: %.8f,\
                                             \nloss_perturb: %.8f, loss_adv: %.8f, \n"
                  "loss_r_adv: %.8f, loss_l2_r_pert: %.8f" %
                  (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                   loss_perturb_sum / num_batch, loss_adv_sum / num_batch,
                   loss_r_adv_sum / num_batch, loss_l2_r_pert_sum / num_batch))

            if epoch % 50 == 0:
                # print("epoch %d:\nloss_D: %.8f, loss_G_fake: %.8f,\
                #                  \nloss_perturb: %.8f, loss_adv: %.8f, \n"
                #       "loss_r_adv: %.8f, loss_l2_r_pert: %.8f, loss_perceptual: %.8f" %
                #       (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                #        loss_perturb_sum / num_batch, loss_adv_sum / num_batch,
                #        loss_r_adv_sum / num_batch, loss_l2_r_pert_sum / num_batch,
                #        loss_perceptual_sum / num_batch))
                netG_file_name = self.models_path + 'netG_epoch_' + str(epoch) + '_1.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

                netR_file_name = self.models_path + 'netR_epoch_' + str(epoch) + '_1.pth'
                torch.save(self.netR.state_dict(), netR_file_name)
            # time_end = time.time()
            # print('time, ', time_end-time_start)

        f = open(self.models_path + 'loss1.txt', 'a', encoding='utf-8')
        f.write('loss_D:' + str(loss_D) + '\n')
        f.write('loss_G_fake:' + str(loss_G_fake) + '\n')
        f.write('loss_perturb:' + str(loss_perturb) + '\n')
        f.write('loss_adv:' + str(loss_adv) + '\n')
        f.write('loss_r_adv:' + str(loss_r_adv) + '\n')
        f.write('loss_l2_r_adv:' + str(loss_l2_r_pert) + '\n')
        # f.write('loss_perceptual:' + str(loss_perceptual) + '\n')
        f.close()

        draw(self.models_path, loss_D, label="D", )
        draw(self.models_path, loss_G_fake, label="G_fake",)
        draw(self.models_path, loss_perturb, label="perturb")
        draw(self.models_path, loss_adv, label="adv")
        draw(self.models_path, loss_r_adv, label="r_adv")
        draw(self.models_path, loss_l2_r_pert, label="l2_r_pert")
        # draw(self.models_path, loss_perceptual, label="perceptual")
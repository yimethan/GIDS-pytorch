from model.discriminator import Discriminator
from model.generator import Generator
from config import Config

import torch
import torch.nn as nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
import os
import numpy as np
import time

writer = SummaryWriter()

device = torch.device('cuda')

gen = Generator()
dis = Discriminator()

criterion = nn.BCELoss()

gen.cuda()
dis.cuda()
criterion.cuda()
Tensor = torch.cuda.FloatTensor

dataset = ImageFolder(root=Config.dataset_path, transform=transforms.Compose([
    transforms.Grayscale(1), transforms.ToTensor()
]))

dataloader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=False, num_workers=1)

optimizer = Adam()

optim_G = Adam(gen.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optim_D = Adam(dis.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

total_steps = len(dataloader) // Config.batch_size * Config.epochs

step = 0

def log_time(batch_idx, duration, g_loss, d_loss):

    samples_per_sec = Config.batch_size / duration
    training_t_left = (total_steps / step - 1.0) * duration if step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                   " | Gen_loss: {:.5f} | Dis_loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(epoch, batch_idx, samples_per_sec, g_loss, d_loss,
                              sec_to_hm_str(duration), sec_to_hm_str(training_t_left)))

def sec_to_hm_str(t):
    # 10239 -> '02h50m39s'

    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)

real_label, fake_label = 1, 0

for epoch in range(Config.epochs):

    for batch_idx, (inputs, _) in enumerate(dataloader):

        start_time = time.time()

        # TODO: train discriminator for real data

        dis.zero_grad()

        inputs = Tensor(inputs)
        label = torch.full((inputs.size(),), real_label, dtype=torch.float, device='cuda')

        output = dis(inputs).view(-1)

        dis_real_loss = criterion(output, label)
        dis_real_loss.backward()

        # TODO: train discriminator for fake data

        noise = torch.randn(inputs.size(0), out=256, dtype=1, layout=1, device='cuda')

        fake_inputs = gen(noise)
        label.fill_(fake_label)

        output = dis(fake_inputs)

        dis_fake_loss = criterion(output, label)
        dis_fake_loss.backward()

        dis_total_loss = dis_real_loss + dis_fake_loss

        writer.add_scalar('loss/dis_total_loss', dis_total_loss.data. epoch)

        optim_D.step()

        # TODO: train generator

        gen.zero_grad()

        label.fill_(real_label)
        output = dis(fake_inputs).view(-1)

        gen_loss = criterion(output, label)
        gen_loss.backward()

        optim_G.step()

        gen_path = Config.save_path + '/gen/epoch_{}'.format(epoch)
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        dis_path = Config.save_path + '/dis/epoch_{}'.format(epoch)
        if not os.path.exists(dis_path):
            os.makedirs(dis_path)

        torch.save(gen.state_dict(), gen_path + '/state_dict.pth')
        torch.save(dis.state_dict(), dis_path + '/state_dict.pth')
        torch.save(gen, gen_path + '/model.pth')
        torch.save(dis, gen_path + '/model.pth')

        duration = time.time() - start_time

        if batch_idx % 100 == 0:
            print("[Train] Epoch: {}/{}, Batch: {}/{}, D loss: {}, G loss: {}".format(epoch, Config.epochs,
                                                                              batch_idx, len(dataloader),
                                                                              dis_total_loss, gen_loss))

        if batch_idx % Config.log_f == 0:
            log_time(batch_idx, duration, gen_loss.cpu().data, dis_total_loss.cpu().data)

        step += 1

        with torch.no_grad():

            img_path = Config.save_path + '/sample_images/{}'.format(epoch)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            random_x = torch.randn(64, 256, 1, 1).to('cuda')
            test_sample = gen(random_x).detach().cpu()

            save_image(test_sample, '{}/{}.jpg'.format(img_path, epoch))
            writer.add_image('sample_imgs', test_sample, epoch)
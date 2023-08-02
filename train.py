from model.discriminator import Discriminator
from model.generator import Generator
from config import Config
from load_dataset import Dataset

import torch
import torch.nn as nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader, random_split
import os
import time

writer = SummaryWriter()
compute_acc = BinaryAccuracy(threshold=Config.detection_thr)

device = torch.device('cuda')

gen = Generator()
dis1 = Discriminator()
dis2 = Discriminator()

criterion = nn.BCELoss()

gen.cuda()
dis1.cuda()
dis2.cuda()
criterion.cuda()
Tensor = torch.cuda.FloatTensor

dataset = Dataset()

train_size = len(dataset) * 0.9
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=False, num_workers=1)
test_dataloader = DataLoader(dataset=test_set, batch_size=Config.batch_size, shuffle=False, num_workers=1)

optim_G = Adam(gen.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optim_D1 = Adam(dis1.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optim_D2 = Adam(dis2.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

total_steps = len(train_dataloader) // Config.batch_size * Config.epochs

step = 0


def log_time(batch_idx, duration, g_loss, d1_loss, d2_loss):
    samples_per_sec = Config.batch_size / duration
    training_t_left = (total_steps / step - 1.0) * duration if step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                   " | Gen_loss: {:.5f} | Dis1_loss: {:.5f} | Dis2_loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(epoch, batch_idx, samples_per_sec, g_loss, d1_loss, d2_loss,
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
normal_label, abnormal_label = 1, 0

for epoch in range(Config.epochs):

    # TODO: TRAIN

    dis1.train()
    dis2.train()
    gen.train()

    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):

        # TODO: train first discriminator for normal/abnormal data

        dis1.zero_grad()

        inputs = Tensor(inputs)

        output = dis1(inputs).view(-1)

        dis_1_loss = criterion(output, labels)
        dis_1_loss.backward()

        writer.add_scalar('loss/dis1_loss', dis_1_loss.data.epoch)

        # TODO: train second discriminator for real data

        dis2.zero_grad()

        output = dis2(inputs).view(-1)

        labels.fill_(real_label)

        dis_2_real_loss = criterion(output, labels)
        dis_2_real_loss.backward()

        # TODO: train second discriminator for fake data

        noise = torch.randn(inputs.size(0), out=256, dtype=1, layout=1, device='cuda')

        fake_inputs = gen(noise)
        labels.fill_(fake_label)

        output = dis2(fake_inputs)

        dis_2_fake_loss = criterion(output, labels)
        dis_2_fake_loss.backward()

        dis_2_total_loss = dis_2_real_loss + dis_2_fake_loss

        writer.add_scalar('loss/dis2_total_loss', dis_2_total_loss.data.epoch)

        optim_D2.step()

        # TODO: train generator

        gen.zero_grad()

        labels.fill_(real_label)
        output = dis2(fake_inputs).view(-1)

        gen_loss = criterion(output, labels)
        gen_loss.backward()

        optim_G.step()

        # TODO: save checkpoints

        gen_path = Config.save_path + '/gen/epoch_{}'.format(epoch)
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        dis1_path = Config.save_path + '/dis1/epoch_{}'.format(epoch)
        if not os.path.exists(dis1_path):
            os.makedirs(dis1_path)
        dis2_path = Config.save_path + '/dis2/epoch_{}'.format(epoch)
        if not os.path.exists(dis2_path):
            os.makedirs(dis2_path)

        torch.save(gen.state_dict(), gen_path + '/state_dict.pth')
        torch.save(dis1.state_dict(), dis1_path + '/state_dict.pth')
        torch.save(dis2.state_dict(), dis2_path + '/state_dict.pth')
        torch.save(gen, gen_path + '/model.pth')
        torch.save(dis1, dis1_path + '/model.pth')
        torch.save(dis2, dis2_path + '/model.pth')

        duration = time.time() - start_time

        if batch_idx % 100 == 0:
            print("[Train] Epoch: {}/{}, Batch: {}/{}, D_1 loss: {}, d_2 loss: {}, G loss: {}".format(epoch,
                                                                                                      Config.epochs,
                                                                                                      batch_idx,
                                                                                                      len(train_dataloader),
                                                                                                      dis_1_loss,
                                                                                                      dis_2_total_loss,
                                                                                                      gen_loss))

        if batch_idx % Config.log_f == 0:
            log_time(batch_idx, duration, gen_loss.cpu().data, dis_1_loss.cpu().data, dis_2_total_loss.cpu().data)

        step += 1

    # TODO: TEST

    dis1.eval()
    dis2.eval()
    gen.eval()

    with torch.no_grad():

        img_path = Config.save_path + '/generated_img_samples'.format(epoch)
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        random_x = torch.randn(64, 256, 1, 1).to('cuda')
        test_sample = gen(random_x).detach().cpu()

        save_image(test_sample, '{}/{}.png'.format(img_path, epoch))
        writer.add_image('generated_img_samples', test_sample, epoch)

        batch_acc = 0

        for batch_idx, (inputs, labels) in enumerate(test_dataloader):

            output = dis1(inputs)

            batch_acc += compute_acc(output, labels)

        epoch_acc = batch_acc / len(test_dataloader)
        writer.add_scalar('test_acc', epoch_acc, epoch)
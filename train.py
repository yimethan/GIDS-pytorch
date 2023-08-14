from config import Config
from load_dataset import Dataset
from model.discriminator import Discriminator
from model.generator import Generator

import torch
import torch.nn as nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader, random_split
import os
import time

device = torch.device('cuda')

compute_acc = BinaryAccuracy(threshold=Config.detection_thr).to(device)

# TODO: cannot train multiple models simultaneously?
gen = Generator().to(device)
dis1 = Discriminator().to(device)
dis2 = Discriminator().to(device)

criterion = nn.BCELoss()

dataset = Dataset()

writer = SummaryWriter()

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=False, num_workers=1)
test_dataloader = DataLoader(dataset=test_set, batch_size=Config.batch_size, shuffle=False, num_workers=1)

optim_G = Adam(gen.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optim_D1 = Adam(dis1.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optim_D2 = Adam(dis2.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

total_steps = len(train_dataloader) // Config.batch_size * Config.epochs

step = 0

def log_time(epoch, batch_idx, duration, g_loss, d1_loss, d2_loss):
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


real_label, fake_label = 0, 1
normal_label, abnormal_label = 0, 1


def train():
    global step

    for epoch in range(Config.epochs):

        # TODO: TRAIN

        dis1.train()
        dis2.train()
        gen.train()

        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):

            # TODO: train first discriminator for normal/abnormal data

            dis1.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            # print(inputs.size())  # batch, 1, 64, 48

            output = dis1(inputs).to(device)

            # print(output.size())  # 16
            # print(labels.size())  # 16

            dis_1_loss = criterion(output.to(torch.float32), labels.to(torch.float32))
            dis_1_loss.backward()

            writer.add_scalar('loss/dis1_loss', dis_1_loss.data, epoch)

            # TODO: train second discriminator for real data

            dis2.zero_grad()

            output = dis2(inputs)

            labels.fill_(real_label)

            dis_2_real_loss = criterion(output.to(torch.float32), labels.to(torch.float32))
            dis_2_real_loss.backward()

            # TODO: train second discriminator for fake data

            noise = torch.randn(Config.batch_size, 256, 1, 1).to(device)
            # print(noise.size())

            fake_inputs = gen(noise).to(device)
            labels.fill_(fake_label)
            # print(fake_inputs.size(), labels.size())

            output = dis2(fake_inputs)

            try:
                dis_2_fake_loss = criterion(output.to(torch.float32), labels.to(torch.float32))
            except ValueError:
                # print('ValueError batch idx:', batch_idx)  # batch 14562
                # print(inputs, inputs.size())
                # print(labels, labels.size())
                # print(output, output.size())
                labels = torch.ones(Config.batch_size).to(device)
                dis_2_fake_loss = criterion(output.to(torch.float32), labels.to(torch.float32))

            dis_2_fake_loss.backward(retain_graph=True)

            dis_2_total_loss = dis_2_real_loss + dis_2_fake_loss

            writer.add_scalar('loss/dis2_total_loss', dis_2_total_loss.data, epoch)

            optim_D2.step()

            # TODO: train generator

            gen.zero_grad()

            labels.fill_(real_label)

            gen_loss = criterion(output.to(torch.float32), labels.to(torch.float32))
            gen_loss.backward()

            writer.add_scalar('loss/gen_total_loss', gen_loss.data, epoch)

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

            if batch_idx % Config.log_f == 0:
                print("[Train] Epoch: {}/{}, Batch: {}/{}, D_1 loss: {}, d_2 loss: {}, G loss: {}".format(epoch,
                                                                                                          Config.epochs,
                                                                                                          batch_idx,
                                                                                                          len(train_dataloader),
                                                                                                          dis_1_loss,
                                                                                                          dis_2_total_loss,
                                                                                                          gen_loss))

            step += 1

        log_time(epoch, batch_idx, duration, gen_loss.cpu().data, dis_1_loss.cpu().data, dis_2_total_loss.cpu().data)

        # TODO: TEST

        dis1.eval()
        dis2.eval()
        gen.eval()

        with torch.no_grad():

            img_path = Config.save_path + '/generated_img_samples'.format(epoch)
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            random_x = torch.randn(64, 256, 1, 1).to(device)
            test_sample = gen(random_x).detach().cpu()

            save_image(test_sample, '{}/{}.png'.format(img_path, epoch))
            writer.add_image('generated_img_samples', test_sample, epoch, dataformats='NCHW')

            batch_acc = 0

            for batch_idx, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # print('inputs:', inputs.size())
                # print('labels:', labels.size())

                output = dis1(inputs).to(device)
                # print('output (right after dis1):', output, output.size())

                for out in range(len(output)):
                    try:
                        if output[out] < Config.detection_thr:
                            output[out] = dis2(inputs)[out].to(device)
                    except IndexError:
                        output = output.repeat(8)
                        # print('output (in except):', output, output.size())

                        if output[out] < Config.detection_thr:
                            output[out] = dis2(inputs)[out].to(device)

                output = output.to(device)

                batch_acc += compute_acc(output.to(torch.float32), labels.to(torch.float32))

            epoch_acc = batch_acc / len(test_dataloader)
            print(f'{epoch} test accuracy: {epoch_acc}')

            writer.add_scalar('test_acc', epoch_acc, epoch)


if __name__ == '__main__':
    train()

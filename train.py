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

device = torch.device('cuda')

compute_acc = BinaryAccuracy(threshold=Config.detection_thr).to(device)

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

real_label, fake_label = 0, 1
normal_label, abnormal_label = 0, 1


def train():

    for epoch in range(Config.epochs):

        # TODO: TRAIN

        dis1.train()
        dis2.train()
        gen.train()

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):

            optim_G.zero_grad()
            optim_D1.zero_grad()
            optim_D2.zero_grad()

            inputs = inputs.to(device)  # batch, 1, 64, 48
            labels = labels.to(device)

            # TODO: train generator

            # labels.fill_(real_label)
            gen_target = torch.zeros(Config.batch_size, requires_grad=False).to(device)

            noise = torch.randn(Config.batch_size, 256, 1, 1).to(device)
            fake_inputs = gen(noise).to(device)

            gen_loss = criterion(dis2(fake_inputs).to(torch.float32), gen_target.to(torch.float32))
            gen_loss.backward()
            optim_G.step()

            # TODO: train first discriminator for normal/abnormal data

            dis1_output = dis1(inputs).to(device)

            dis_1_loss = criterion(dis1_output.to(torch.float32), labels.to(torch.float32))
            dis_1_loss.backward()
            optim_D1.step()

            # TODO: train second discriminator for real/fake data

            # noise = torch.randn(Config.batch_size, 256, 1, 1).to(device)
            # fake_inputs = gen(noise).to(device)

            dis2_real_output = dis2(inputs).to(device)
            real_target = torch.zeros(dis2_real_output.shape[0], requires_grad=False).to(device)

            dis_2_real_loss = criterion(dis2_real_output.to(torch.float32), real_target.to(torch.float32))
            # dis_2_real_loss.backward()

            dis2_fake_output = dis2(fake_inputs.detach())
            fake_target = torch.ones(dis2_fake_output.shape[0], requires_grad=False).to(device)

            dis_2_fake_loss = criterion(dis2_fake_output.to(torch.float32), fake_target.to(torch.float32))
            # dis_2_fake_loss.backward()

            dis_2_total_loss = (dis_2_real_loss + dis_2_fake_loss) / 2
            dis_2_total_loss.backward()

            optim_D2.step()

            # TODO: save checkpoints

            writer.add_scalar('loss/dis1_loss', dis_1_loss.data, epoch)

            writer.add_scalar('loss/dis_2_real_loss', dis_2_real_loss.data, epoch)
            writer.add_scalar('loss/dis2_fake_loss', dis_2_fake_loss, epoch)
            writer.add_scalar('loss/dis2_total_loss', dis_2_total_loss, epoch)

            writer.add_scalar('loss/gen_loss', gen_loss.data, epoch)

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

            if batch_idx % Config.log_f == 0:
                print("[Train] Epoch: {}/{}, Batch: {}/{}, D_1 loss: {}, d_2 loss: {}, G loss: {}".format(epoch,
                           Config.epochs, batch_idx, len(train_dataloader), dis_1_loss, dis_2_total_loss, gen_loss))


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

            save_image(test_sample[0], '{}/{}.png'.format(img_path, epoch))
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
                    if output[out] < Config.detection_thr:
                        output[out] = dis2(inputs)[out].to(device)

                output = output.to(device)

                batch_acc += compute_acc(output.to(torch.float32), labels.to(torch.float32))

            epoch_acc = batch_acc / len(test_dataloader)
            print(f'{epoch} test accuracy: {epoch_acc}')

            writer.add_scalar('test_acc', epoch_acc, epoch)


if __name__ == '__main__':
    train()
    writer.export_scalars_to_json(Config.save_path + '/scalars.json')
    writer.close()

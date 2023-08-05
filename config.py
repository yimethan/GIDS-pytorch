import torch

class Config:

    epochs = 30
    batch_size = 16
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999

    detection_thr = 0.1
    input_size = (256, 6, 4)
    resize_size = (512, 3, 2)
    dis_input_size = (1, 16, )

    sample_f = 1
    log_f = 125

    save_path = './ckpt'
    dataset_path = '../dataset/CHD/id_image/'
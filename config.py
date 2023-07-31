import torch

class Config:

    epochs = 10
    batch_size = 32
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    channels = 1

    detection_thr = 0.1
    attack_thr = 1
    input_size = 64

    sample_f = 1
    log_f = 125

    save_path = './ckpt'
    dataset_path = '../CHD/converted_to_img/'
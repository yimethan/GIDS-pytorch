class Config:

    epochs = 10
    batch_size = 64
    lr = 0.0001
    b1 = 0.5
    b2 = 0.999

    detection_thr = 0.1
    resize_size = (512, 3, 2)
    dis_input_size = (batch_size, 1, 64, 48)

    log_f = 3000

    save_path = './ckpt'
    dataset_path = '../dataset/CHD/id_image_64/'

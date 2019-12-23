from datasets.DataFactory import get_data_loader

data_root = "/root/data/DFDC/fb_dfd_release_0.1_final"
category = "heads_13_RetinaFace"
dataloader = get_data_loader(data_root, category, batch_size=16, num_workers=2, input_size=224, interval=10, landmarks=False, mode='train')
for imgs, labels in dataloader:
    print(imgs.size())
    print(labels.size())
    break

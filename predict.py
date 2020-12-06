import torchvision.transforms as transforms
import torch
from dataLoader import MyDataset
from torch.utils.data import DataLoader
from Net import UNet
import argparse
import cv2
import os


def eval_net(net, testdataLoader, device, batch):
    net.eval()
    for imgs, names in testdataLoader:
        imgs = imgs.to(device='cuda')
        with torch.no_grad():
            output = net(imgs)
            for index in range(batch):
                img_pred = output
                img_pred = img_pred[index].permute(1, 2, 0)
                img_pred.cpu().numpy()
                img_pred = img_pred * 255
                cv2.imwrite(os.path.join('./pre', names[index]), img_pred)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    parser = argparse.ArgumentParser(description='UNet param')
    parser.add_argument('--test_image_dir', default=r'./data/test')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    datasets = MyDataset(args.test_image_dir, train=False)
    testdataLoader = DataLoader(datasets, batch_size=args.batch_size)

    checkpoint = torch.load("UNet.pt")

    net = UNet(1, 1).cuda()
    net.load_state_dict(torch.load('UNet.pt'))

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    eval_net(net, testdataLoader, device=device, batch=args.batch_size)
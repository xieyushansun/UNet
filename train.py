import argparse
import torch
import torch.nn as nn
from torch import optim
from dataLoader import MyDataset
from torch.utils.data import DataLoader
from Net import UNet


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train_Epoch:{} [{}/{} ({:.2f}%)] \t loss:{:.6f}'
                  .format(epoch, batch_idx * len(data), len(train_loader),
                          100.0 * batch_idx / len(train_loader), loss.item()))


def main():
    parser = argparse.ArgumentParser(description='UNet param')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_dir', default='./data/train/image')
    parser.add_argument('--label_dir', default='./data/train/label')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    # 加载训练数据
    datasets = MyDataset(args.img_dir, args.label_dir, True)
    train_loader = DataLoader(datasets, batch_size=args.batch_size)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, args.epochs + 1):
        train(args, net, device, train_loader, optimizer, criterion, epoch)
    torch.save(net.state_dict(), "UNet.pt")


if __name__ == '__main__':
    main()
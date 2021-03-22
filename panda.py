import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    features_space = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            images = imgs.to(device)
            _, features = model(images)
            batch_size = features.shape[0]
            for j in range(batch_size):
                features_space.append(features[j].detach().cpu().numpy())
    train_set = np.array(features_space)
    center = torch.FloatTensor(train_set).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader):
    features_space = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            _, features = model(imgs)
            batch_size = features.shape[0]
            for j in range(batch_size):
                features_space.append(features[j].detach().cpu().numpy())
    train_set = np.array(features_space)
    anom_labels = []
    features_space = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.numpy()
            _, features = model(imgs)
            batch_size = features.shape[0]
            for j in range(batch_size):
                features_space.append(features[j].detach().cpu().numpy())
                anom_labels.append(labels[j])
    test_set = np.array(features_space)
    test_labels = np.array(anom_labels)

    distances = utils.knn_score(train_set, test_set)

    auc = roc_auc_score(test_labels, distances)

    return auc


def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    train_model(model, train_loader, test_loader, device, args, ewc_loss)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--all', action='store_true', help='Get full experiment results. ONLY FOR CIFAR10/FMNIST')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    main(args)

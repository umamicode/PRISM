import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

#ReLIC
#[TODO]
from relic import ReLIC
#from relic.modules import ReLIC_Loss, get_resnet
from relic.modules.transformations import TransformsRelic

# TensorBoard
#from torch.utils.tensorboard import SummaryWriter


from utils import yaml_config_hook


def inference(loader, ssl_model, device ,relic):
    feature_vector = []
    labels_vector = []
    #SIMCLR
    if relic == False:
        for step, (x, y) in enumerate(loader):
            x = x.to(device)

            # get encoding
            with torch.no_grad():
                h, _, z, _ = ssl_model(x, x)

            h = h.detach()

            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.numpy())

            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector
    #ReLIC
    elif relic == True:
        for step, (x, y) in enumerate(loader):
            x = x.to(device)

            # get encoding
            with torch.no_grad():
                h,_,_, _, _, _, _ = ssl_model(x, x, x)
                #online_1,online_2,target_1,target_2, original_features = ssl_model(x, x, x)
                #print("online1 -- ", online_1.shape)
                #print("online2 -- ", online_2.shape)
                #print("target1 -- ", target_1.shape)
                #print("target2 -- ", target_2.shape)
                #print("original -- ", original_features.shape)
                #print("AAAAAAAAAAAA")
                


            
            h = h.detach() #(256,512)

            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.numpy())

            

            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector        

def get_features(ssl_model, train_loader, test_loader, device, relic):
    train_X, train_y = inference(train_loader, ssl_model, device, relic) #relic
    test_X, test_y = inference(test_loader, ssl_model, device, relic) #relic
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, ssl_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)
        

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, ssl_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

def main(gpu,args):

    print("ReLIC -- {c}".format(c= args.relic))
    print("Model From -- {m}, Epoch: {e}".format(m= args.model_path, e= args.epoch_num))
    print("All Evaluation args -- ", args)

    
    if args.relic == False:
        if args.test_dataset == "STL10":
            train_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="test",
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        elif args.test_dataset == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=False,
                download=True,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
            )
        else:
            raise NotImplementedError
    
    #[TODO - Added] ReLIC
    elif args.relic == True:
        if args.test_dataset == "STL10":
            train_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="train",
                download=True,
                transform=TransformsRelic(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="test",
                download=True,
                transform=TransformsRelic(size=args.image_size).test_transform,
            )
        elif args.test_dataset == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=True,
                download=True,
                transform=TransformsRelic(size=args.image_size).test_transform,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                train=False,
                download=True,
                transform=TransformsRelic(size=args.image_size).test_transform,
            )
        else:
            raise NotImplementedError       


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    if args.relic ==  False:
        simclr_model = SimCLR(encoder, args.projection_dim, n_features)
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        simclr_model = simclr_model.to(args.device)
        simclr_model.eval()

        ## Logistic Regression
        n_classes = 10  # CIFAR-10 / STL-10
        model = LogisticRegression(simclr_model.n_features, n_classes)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        print("### Creating features from pre-trained context model ###")
        (train_X, train_y, test_X, test_y) = get_features(
            simclr_model, train_loader, test_loader, args.device, args.relic
        )

        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, args.logistic_batch_size
        )

        for epoch in range(args.logistic_epochs):
            loss_epoch, accuracy_epoch = train(
                args, arr_train_loader, simclr_model, model, criterion, optimizer
            )
            print(
                f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
            )

        # final testing
        loss_epoch, accuracy_epoch = test(
            args, arr_test_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
        )

    #[TODO - ADDED] ReLIC
    if args.relic == True:
        relic_model = ReLIC(encoder, args.projection_dim, n_features)
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        relic_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

        #[TODO] JUST USE ENCODER PART ###CAUTION
        #saved_n_features= relic_model.n_features
        #relic_model= relic_model.encoder

        relic_model = relic_model.to(args.device)
        relic_model.eval()

        

        ## Logistic Regression
        n_classes = 10  # CIFAR-10 / STL-10
        #[TODO] JUST USE ENCODER PART ### CAUTION / OR USE Smaller Model (args.projection_dim)
        #model = LogisticRegression(args.projection_dim, n_classes)
        
        model = LogisticRegression(relic_model.n_features, n_classes) #(relic_model.n_features, n_classes)= (512,10)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        print("### Creating features from pre-trained context model ###")
        (train_X, train_y, test_X, test_y) = get_features(
            relic_model, train_loader, test_loader, args.device, args.relic
        )

        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, args.logistic_batch_size
        )

        for epoch in range(args.logistic_epochs):
            loss_epoch, accuracy_epoch = train(
                args, arr_train_loader, relic_model, model, criterion, optimizer
            )
            print(
                f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
            )

        # final testing
        loss_epoch, accuracy_epoch = test(
            args, arr_test_loader, relic_model, model, criterion, optimizer
        )
        print(
            f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
        )











if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation") #(default: SimCLR)
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    print("All Evaluation Args --", args)

    
    
    print("This Part is Working")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(0,args)

    
import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

#ReLIC
#[TODO]
from relic import ReLIC
from relic.modules import ReLIC_Loss, get_resnet
from relic.modules.transformations import TransformsRelic
from relic.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

#import linear_evaluation.py
from subprocess import call 


#PACS Dataset
NUM_CLASSES = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
DATASETS_NAMES = ['photo', 'art', 'cartoon', 'sketch']
CLASSES_NAMES = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
DIR_PHOTO = './datasets/PACS/photo'
DIR_ART = './datasets/PACS/art_painting'
DIR_CARTOON = './datasets/PACS/cartoon'
DIR_SKETCH = './datasets/PACS/sketch'

def train(args, train_loader, model, criterion, optimizer, writer, relic=False):
    loss_epoch = 0
    if relic==False:  #RELIC
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            if args.nr == 0 and step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            if args.nr == 0:
                writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
                args.global_step += 1

            loss_epoch += loss.item()
        return loss_epoch
    
    if relic==True:  #RELIC
            for step, ((x_i, x_j,x_orig), _) in enumerate(train_loader):
                optimizer.zero_grad()
                x_i = x_i.cuda(non_blocking=True)
                x_j = x_j.cuda(non_blocking=True)
                x_orig= x_orig.cuda(non_blocking=True)

                # positive pair, with encoding
                _,_, online_1,online_2,target_1,target_2, original_features = model(x_i, x_j, x_orig)
                loss_1, loss_2 = criterion(online_1, target_2, original_features), criterion(online_2, target_1, original_features)
                loss = loss_1 + loss_2
                #loss = criterion(z_i, z_j)
                loss.backward()

                optimizer.step()

                if dist.is_available() and dist.is_initialized():
                    loss = loss.data.clone()
                    dist.all_reduce(loss.div_(dist.get_world_size()))

                if args.nr == 0 and step % 50 == 0:
                    print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

                if args.nr == 0:
                    writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
                    args.global_step += 1

                loss_epoch += loss.item()
            return loss_epoch

def main(gpu, args):
    ###TEST [TODO- ADDED]

    print("ReLIC -- {relic}".format(relic=args.relic))
    print("Model Saved In -- {m}, Train Epochs: {e}".format(m= args.model_path, e= args.epochs))
    print("All Args --", args)
    


    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #[TODO]Add ReLIC TranformsReLIC is arg.ReLIC=True
    if args.relic== False:
        if args.dataset == "STL10":
            train_dataset = torchvision.datasets.STL10(
                args.dataset_dir,
                split="unlabeled",
                download=True,
                transform=TransformsSimCLR(size=args.image_size),
            )
        elif args.dataset == "CIFAR10":
            train_dataset = torchvision.datasets.CIFAR10(
                args.dataset_dir,
                download=True,
                transform=TransformsSimCLR(size=args.image_size),
            )
        elif args.dataset == "PACS":
                pacs_convertor= {'none':DIR_PHOTO, 'photo':DIR_PHOTO, 'art':DIR_ART, 'cartoon':DIR_CARTOON, 'sketch':DIR_SKETCH}
                train_dataset= torchvision.datasets.ImageFolder(pacs_convertor[args.pacs_style], transform=TransformsSimCLR(size=args.image_size))
                
                '''
                photo_dataset = torchvision.datasets.ImageFolder(DIR_PHOTO, transform=TransformsRelic(size=args.image_size))
                art_dataset = torchvision.datasets.ImageFolder(DIR_ART, transform=TransformsRelic(size=args.image_size))
                cartoon_dataset = torchvision.datasets.ImageFolder(DIR_CARTOON, transform=TransformsRelic(size=args.image_size))
                sketch_dataset = torchvision.datasets.ImageFolder(DIR_SKETCH, transform=TransformsRelic(size=args.image_size))
                '''
        else:
            raise NotImplementedError
    

    #[TODO-ADDED]-ReLIC
    elif args.relic== True:
            if args.dataset == "STL10":
                train_dataset = torchvision.datasets.STL10(
                    args.dataset_dir,
                    split="unlabeled",
                    download=True,
                    transform=TransformsRelic(size=args.image_size),
                )
            elif args.dataset == "CIFAR10":
                train_dataset = torchvision.datasets.CIFAR10(
                    args.dataset_dir,
                    download=True,
                    transform=TransformsRelic(size=args.image_size),
                )
            elif args.dataset == "PACS":
                pacs_convertor= {'default':DIR_PHOTO, 'photo':DIR_PHOTO, 'art':DIR_ART, 'cartoon':DIR_CARTOON, 'sketch':DIR_SKETCH}
                train_dataset= torchvision.datasets.ImageFolder(pacs_convertor[args.pacs_style], transform=TransformsRelic(size=args.image_size))

                '''
                photo_dataset = torchvision.datasets.ImageFolder(DIR_PHOTO, transform=TransformsRelic(size=args.image_size))
                art_dataset = torchvision.datasets.ImageFolder(DIR_ART, transform=TransformsRelic(size=args.image_size))
                cartoon_dataset = torchvision.datasets.ImageFolder(DIR_CARTOON, transform=TransformsRelic(size=args.image_size))
                sketch_dataset = torchvision.datasets.ImageFolder(DIR_SKETCH, transform=TransformsRelic(size=args.image_size))
                '''
            else:
                raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    '''
    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    (REF: https://github.com/robertofranceschi/Domain-adaptation-on-PACS-dataset/blob/master/code/main.py)

    photo_dataloader = DataLoader(photo_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True)
    art_dataloader = DataLoader(art_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=False)
    cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=False)
    sketch_dataloader = DataLoader(sketch_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=False)

    '''


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=args.pretrain) #[MODIFIED- False -> args.pretrain]
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    if args.relic== False:
        model = SimCLR(encoder, args.projection_dim, n_features)
    elif args.relic== True:
        #[TODO]
        model= ReLIC(encoder, args.projection_dim, n_features)
        

    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)

    # [TODO] Add ReLIC Objective ici.
    if args.relic== False:
        criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
    #[TODO- Added] Add ReLic Loss
    if args.relic==True:   
       criterion = ReLIC_Loss(args.relic_normalize, args.relic_temp, args.relic_alpha)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer, relic=args.relic)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR/ReLIC")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)

    call(["python3", "linear_evaluation.py"])
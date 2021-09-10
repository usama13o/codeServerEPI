from typing import Sized
from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj

from models import get_model
import numpy as np
import os
from utils.metrics import dice_score, distance_metric, precision_and_recall
from utils.error_logger import StatLogger


def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def validation(args):
    # Load options
    json_opts = json_file_to_pyobj(args.config,args=args)
    train_opts = json_opts.training

    # Setup the NN Model
    model = get_model(json_opts.model)
    save_directory = os.path.join(model.save_dir, train_opts.arch_type); mkdirfun(save_directory)

    # Setup Dataset and Augmentation
    dataset_class = get_dataset(train_opts.arch_type)
    dataset_path = get_dataset_path(train_opts.arch_type, json_opts.data_path)
    dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)

    # Setup Data Loader
    dataset = dataset_class(dataset_path, split='test', transform=dataset_transform['valid'])
    data_loader = DataLoader(dataset=dataset, num_workers=4, batch_size=1, shuffle=False)

    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)

    # Setup stats logger
    stat_logger = StatLogger()

    # test
    for iteration, data in enumerate(data_loader, 1):
        model.set_input(data[0])
        model.test()

        input_arr  = np.squeeze(data[0].cpu().squeeze().numpy()).astype(np.uint8)
        gt_arr= np.squeeze(data[1].cpu().squeeze().numpy()).astype(np.uint8)
        output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)
        if len(np.unique(output_arr))> 2:
            output_arr[output_arr==1]=100
            output_arr[output_arr==2]=190
            output_arr[output_arr==3]=250
        else:
            output_arr[output_arr==1]=255
        from PIL.Image import fromarray
        if os.path.exists(f"./output_preds/{json_opts.visualisation.run_name}") is False:
            os.mkdir(f"./output_preds/{json_opts.visualisation.run_name}")
        fromarray(output_arr).convert("L").save(f"./output_preds/{json_opts.visualisation.run_name}/{iteration}.png")
        fromarray(input_arr).save(f"./output_preds/{json_opts.visualisation.run_name}/_{iteration}.png")
        fromarray(gt_arr*255).convert("L").save(f"./output_preds/{json_opts.visualisation.run_name}/gt_{iteration}.png")




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Validation Function')


    parser.add_argument('-c', '--config',  help='training config file', required=False)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('-a', '--arch_type',   help='wich architecture type')
    parser.add_argument('-wandb', '--use_wandb',   help='use wandb to log the training',type=bool,default=False)
    parser.add_argument('-cont', '--continue_train',   help='Should contine training?',type=bool,default=False)
    parser.add_argument('-wep', '--which_epoch',   help='which epoch to continue training from?')
    parser.add_argument('-maxlr', '--max_lr',   help='maximum learning rate for cyclic learning')
    parser.add_argument('-bs', '--batchSize',   help='batch size',type=int)
    parser.add_argument('-img', '--img_size',   help='batch size',type=int)
    parser.add_argument('-ep', '--n_epochs',   help='number of epochs', type=int)
    parser.add_argument('-pre', '--path_pre_trained_model',   help='path to pre trained model')
    args = parser.parse_args()

    validation(args)

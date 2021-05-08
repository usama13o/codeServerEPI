import pywick 
from pywick.models.segmentation import deeplab_v3_plus
from utils.util import json_file_to_pyobj
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from torch.utils.data import DataLoader
from pywick.modules import ModuleTrainer
import pywick.metrics as pwm
from utils.error_logger import ErrorLogger

import numpy
from tqdm import tqdm


from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model



# Parse input arguments
json_filename = "configs/config_TransUnet.json"

# Load options
json_opts = json_file_to_pyobj(json_filename)
train_opts = json_opts.training

# Architecture type
arch_type = train_opts.arch_type
model = get_model(json_opts.model)

# Setup Dataset and Augmentation
ds_class = get_dataset(arch_type)
ds_path  = get_dataset_path(arch_type, json_opts.data_path)
ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

# Setup Data Loader
train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
train_loader = DataLoader(dataset=train_dataset, num_workers=7, batch_size=train_opts.batchSize, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, num_workers=1, batch_size=train_opts.batchSize, shuffle=False)
test_loader  = DataLoader(dataset=test_dataset,  num_workers=0, batch_size=train_opts.batchSize, shuffle=False)

# metrics = [pwm.DiceCoefficientMetric(is_binary=False)]
# trainer = ModuleTrainer(model)
visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir,resume= True if json_opts.model.continue_train else False)
error_logger = ErrorLogger()
start_epoch = False if json_opts.training.n_epochs < json_opts.model.which_epoch else json_opts.model.continue_train
model.set_scheduler(train_opts,len_train=len(train_loader),max_lr=json_opts.model.max_lr,division_factor=json_opts.model.division_factor,last_epoch=json_opts.model.which_epoch * len(train_loader) if start_epoch else -1)

for epoch in range(0, train_opts.n_epochs):
    print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))
    if epoch == 3:
        print("freezing model")
        model.freeze()

    # Training Iterations
    for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
        # Make a training update
        model.set_input(images, labels)
        # model.optimize_parameters()
        model.optimize_parameters_accumulate_grd(epoch_iter)
        lr = model.update_learning_rate()
        lr = {"lr":lr}
        # Error visualisation
        errors = model.get_current_errors()
        stats = model.get_segmentation_stats()
        error_logger.update({**errors, **stats,**lr}, split='train')
        visualizer.plot_current_errors(epoch, error_logger.get_errors('train'), split_name='train')

    # Validation and Testing Iterations
    for loader, split in zip([valid_loader], ['validation']):
        for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

            # Make a forward pass with the model
            model.set_input(images, labels)
            model.validate()

            # Error visualisation
            errors = model.get_current_errors()
            stats = model.get_segmentation_stats()
            error_logger.update({**errors, **stats}, split=split)

            # Visualise predictions
            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

    # Update the plots
    for split in ['validation']:
        visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
    error_logger.reset()

    # Save the model parameters
    if epoch % train_opts.save_epoch_freq == 0:
        model.save(epoch)
        visualizer.save_model(epoch)

    # Update the model learning rate
    # model.update_learning_rate()



visualizer.finish()

from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from models import get_model
import os, time

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math, numpy
import numpy as np 
# from scipy.misc import imresize
from skimage.transform import resize

def plotNNFilter(units, figure_id, interp='bilinear', colormap=cm.jet, colormap_lim=None, title=''):
    plt.ion()
    filters = units.shape[2]
    n_columns = round(math.sqrt(filters))
    n_rows = math.ceil(filters / n_columns) + 1
    fig = plt.figure(figure_id, figsize=(n_rows*3,n_columns*3))
    fig.clf()

    for i in range(filters):
        ax1 = plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[:,:,i].T, interpolation=interp, cmap=colormap)
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.colorbar()
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.suptitle(title)

def plotNNFilterOverlay(input_im, units, figure_id, interp='bilinear',
                        colormap=cm.jet, colormap_lim=None, title='', alpha=0.8,save=False):
    plt.ion()
    filters = units.shape[2] if len(units.shape) > 2 else 1
    fig = plt.figure(figure_id, figsize=(5,5))
    fig.clf()

    for i in range(filters):
        plt.imshow(input_im[:,:,0], interpolation=interp, cmap='gray')
        plt.imshow(units[:,:,i] if filters > 1 else units, interpolation=interp, cmap=colormap, alpha=alpha)
        plt.axis('off')
        plt.colorbar()
        plt.title(title, fontsize='small')
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if save:
        plt.savefig('{}/{}.png'.format(dir_name,time.time()))




## Load options
PAUSE = .01
#config_name = 'config_sononet_attention_fs8_v6.json'
#config_name = 'config_sononet_attention_fs8_v8.json'
#config_name = 'config_sononet_attention_fs8_v9.json'
#config_name = 'config_sononet_attention_fs8_v10.json'
#config_name = 'config_sononet_attention_fs8_v11.json'
#config_name = 'config_sononet_attention_fs8_v13.json'
#config_name = 'config_sononet_attention_fs8_v14.json'
#config_name = 'config_sononet_attention_fs8_v15.json'
#config_name = 'config_sononet_attention_fs8_v16.json'
#config_name = 'config_sononet_grid_attention_fs8_v1.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v1.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v2.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v3.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v4.json'

# config_name = 'config_sononet_grid_att_fs8_avg.json'
config_name = 'config_sononet_grid_att_fs8_avg_v2.json'
# config_name = 'config_sononet_grid_att_fs8_avg_v3.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v4.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v5.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v5.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v6.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v7.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v8.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v9.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v10.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v11.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v12.json'

config_name = 'config_sononet_grid_att_fs8_avg_v12_scratch.json'
config_name = 'att.json'

#config_name = 'config_sononet_grid_attention_fs8_v3.json'

json_opts = json_file_to_pyobj(f'/mnt/data/Other/Projects/codeServerEPI/Attention-Gated-Networks/configs/{config_name}')
train_opts = json_opts.training

dir_name = os.path.join('visualisation_debug', config_name)
os.system(f"rm -r {dir_name}")
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name,'pos'))
    os.makedirs(os.path.join(dir_name,'neg'))


# Setup the NN Model
model = get_model(json_opts.model)
if hasattr(model.net, 'classification_mode'):
    model.net.classification_mode = 'attention'
if hasattr(model.net, 'deep_supervised'):
    model.net.deep_supervised = False 

# Setup Dataset and Augmentation
dataset_class = get_dataset(train_opts.arch_type)
dataset_path = get_dataset_path(train_opts.arch_type, json_opts.data_path)
dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)

# Setup Data Loader
dataset = dataset_class(dataset_path, split='train', transform=dataset_transform['valid'])
data_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

# test
for iteration, data in enumerate(data_loader, 1):
    model.set_input(data[0], data[1])

    cls = int(data[1].max())

    model.validate()
    pred_class = np.transpose(model.pred_seg.cpu().numpy().astype(np.uint8),(0,2,3,1)).squeeze(3).squeeze()
    pred_cls = int(pred_class.max())

    gt = np.transpose(data[1].cpu().squeeze(4).numpy().astype(np.uint8),(0,2,3,1)).squeeze(3).squeeze()
    

    #########################################################
    # Display the input image and Down_sample the input image
    input_img = model.input[0,0].cpu().numpy()
    #input_img = numpy.expand_dims(imresize(input_img, (fmap_size[0], fmap_size[1]), interp='bilinear'), axis=2)
    input_img = numpy.expand_dims(input_img, axis=2)

    # plotNNFilter(input_img, figure_id=0, colormap="gray")
    plotNNFilterOverlay(input_img,pred_class, figure_id=0, interp='bilinear',
                        colormap=cm.jet,save=False)

    chance = np.random.random() < 0.5 if cls == 1 else 1
    if cls != pred_cls:
        plt.savefig('{}/neg/{:03d}.png'.format(dir_name,iteration))
        plotNNFilterOverlay(input_img,gt, figure_id=0, interp='bilinear',
                        colormap=cm.jet,save=False,title='GT')
        plt.savefig('{}/neg/{:03d}_GT.png'.format(dir_name,iteration))
    elif cls == pred_cls and chance:
        plt.savefig('{}/pos/{:03d}.png'.format(dir_name,iteration))
        plotNNFilterOverlay(input_img,gt, figure_id=0, interp='bilinear',
                        colormap=cm.jet,save=False,title='GT')
        plt.savefig('{}/pos/{:03d}_GT.png'.format(dir_name,iteration))
    #########################################################
    # Compatibility Scores overlay with input
    attentions = []
    layer_name = 'attentionblock4'
    for i in [0,1]:
        fmap = model.get_feature_maps(layer_name, upscale=False)
        if not fmap:
            continue
        #if visulising attention blocks
        if  "attention" in layer_name:
            # Output of the attention block
            fmap_0 = fmap[1][0].squeeze().permute(1,2,0).cpu().numpy()
            fmap_size = fmap_0.shape
            # Attention coefficient (b x c x w x h x s)
            attention = fmap[1][1].squeeze().permute(1,2,0).cpu().numpy()
            attention = attention[:, :,i]
            #attention = numpy.expand_dims(resize(attention, (fmap_size[0], fmap_size[1]), mode='constant', preserve_range=True), axis=2)
            attention = resize(attention, (input_img.shape[0], input_img.shape[1]), mode='constant', preserve_range=True)

            # plotNNFilterOverlay(input_img, attention, figure_id=i, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. {}'.format(cls,pred_cls,i), alpha=0.5)
            # plotNNFilterOverlay(input_img,fmap_0[:,:,i], figure_id=i, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat fmap. {}'.format(cls,pred_cls,i), alpha=0.5)
            attentions.append(attention)
        else:
            fmap_0 = fmap[1].squeeze().permute(1,2,0).cpu().numpy()
            attentions = (fmap[0][0].squeeze().cpu().numpy())
            attentions =np.expand_dims(resize(numpy.mean(attentions,0),(input_img.shape[0],input_img.shape[1]),mode='constant',preserve_range=True),axis=0)

    # this save everything , commented because its too un-organized 
    # plotNNFilterOverlay(input_img, numpy.mean(attentions,0), figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all)'.format(cls, pred_cls), alpha=0.5,save=True)
    fmap_0_resized = resize(numpy.mean(fmap_0,2)[:,:],(input_img.shape[0],input_img.shape[1]),mode='constant',preserve_range=True)
    # plotNNFilterOverlay(input_img,fmap_0_resized, figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all_fmap)'.format(cls, pred_cls), alpha=0.5,save=True)

    if cls != pred_cls:
        plotNNFilterOverlay(input_img,fmap_0_resized, figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all_fmap)'.format(cls, pred_cls), alpha=0.5,save=True)
        plt.savefig('{}/neg/{:03d}_FMAP.png'.format(dir_name,iteration))
        plotNNFilterOverlay(input_img, numpy.mean(attentions,0), figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all)'.format(cls, pred_cls), alpha=0.5,save=True)
        plt.savefig('{}/neg/{:03d}_ATT.png'.format(dir_name,iteration))
    elif cls == pred_cls and chance:
        plotNNFilterOverlay(input_img,fmap_0_resized, figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all_fmap)'.format(cls, pred_cls), alpha=0.5,save=True)
        plt.savefig('{}/pos/{:03d}_FMAP.png'.format(dir_name,iteration))
        plotNNFilterOverlay(input_img, numpy.mean(attentions,0), figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all)'.format(cls, pred_cls), alpha=0.5,save=True)
        plt.savefig('{}/pos/{:03d}_ATT.png'.format(dir_name,iteration))
    # Linear embedding g(x)
    # (b, c, h, w)
    #gx = fmap[2].squeeze().permute(1,2,0).cpu().numpy()
    #plotNNFilter(gx, figure_id=3, interp='nearest', colormap=cm.jet)

    # plt.show()
    # plt.pause(PAUSE)

model.destructor()
#if iteration == 1: break

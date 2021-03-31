# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('apt-get install openslide-tools')
get_ipython().system('pip install openslide-python')
get_ipython().system('pip install scipy==1.1.0')

import utils
get_ipython().system('pip install fastai --upgrade')
from fastai import *
from fastai.vision.all import *
from fastai.vision import *
# from fastai.callbacks.hooks import *|
from glob import glob
get_ipython().system('pip install semtorch')
from semtorch import get_segmentation_learner
get_ipython().system('pip install spams')
import os 


# %%
get_ipython().system('pip install fastai --upgrade')
from fastai import *
from fastai.vision.all import *
from fastai.vision import *
# from fastai.callbacks.hooks import *|
from glob import glob

import os 

get_ipython().system('pip install semtorch')
from semtorch import get_segmentation_learner

print("Unzipping data ...")
# %%
get_ipython().system('unzip -q /content/epithelial-cells-ihc.zip')


# %%
get_ipython().system('unzip -q /content/normal-tifff')


print('Merging mask dirs ... ')
# %%
mv  -v /content/mask/* /content/1024_mask/1024/


# %%
original_file_names= sorted(glob("/content/train/*"))
annotated_file_names = sorted(glob("/content/mask/*"))

# %%
#  original_file_names= sorted(glob("../input/epi-data-512-1024/512/512/*"))
# annotated_file_names = sorted(glob("../input/epi-data-512-1024/512_mask/512/*"))
# len(original_file_names),len(annotated_file_names)


# %%
original_file_names= sorted(glob("/content/1024/1024/*"))
annotated_file_names = sorted(glob("/content/1024_mask/1024/*"))
len(original_file_names),len(annotated_file_names)


# %%

normal_orignal = sorted(glob("/content/normal/1024/train/*"))
normal_ann = sorted(glob("/content/normal/1024/mask/*"))
len(normal_ann),len(normal_orignal)


# %%
both_normal_tum = np.concatenate((annotated_file_names,normal_ann))
len(both_normal_tum)


# %%
get_ipython().system('mv -v /content/1024/1024/* /content/normal/1024/train/')


# %%
def get_y(im_f):
  msk_path = os.path.join(os.path.dirname(annotated_file_names[0]),os.path.basename(im_f))
#   msk = np.array(PILMask.create(msk_path))
#   msk[msk==255]=1
  # mask = msk.astype("int8")
  return msk_path 
def get_y_test(fn):
    base=os.path.basename(fn)
    base=os.path.splitext(base)[0]
    match = [s for s in both_normal_tum if base in s]
    return match[0]
def n_codes(fnames, is_partial=True):
  "Gather the codes from a list of `fnames`"
  vals = set()
  if is_partial:
    random.shuffle(fnames)
    fnames = fnames[:10]
  for fname in fnames:
    msk = np.array(PILMask.create(fname))
    for val in np.unique(msk):
      if val not in vals:
        vals.add(val)
  vals = list(vals)
  p2c = dict()
  for i,val in enumerate(vals):
    p2c[i] = vals[i]
  return p2c  
def get_msk(fn, p2c):
  "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
  fn = get_y(fn)
  msk = np.array(PILMask.create(fn))
  mx = np.max(msk)
  
  for i, val in enumerate(p2c):
    msk[msk==p2c[i]] = val
  print(np.unique(msk))
  msk=msk.astype("bool")
  return PILMask.create(msk).convert("1")
get_yy = lambda o: get_msk(o, n_codes(annotated_file_names))

# %% [markdown]
# OLD WAY 

# %%
vals = n_codes(annotated_file_names)
print(vals)
for i, n in enumerate(vals):
  print(i,n,vals[i])


# %%
# for idx,fn in enumerate(original_file_names):
#     print(f"Done %d of %d"%(idx,len(original_file_names)))
#     ann = get_y(fn)
#     org = PILImage.create(fn)
#     ann = PILImage.create(ann) 
#     print(np.unique(np.asarray(ann)))
#     if 29 in np.unique(np.asarray(ann)): 
#         print(fn)


# %%
im_f = original_file_names[0]
img = PILImage.create(im_f)
img.show()

# %% [markdown]
# 

# %%
ms = (np.array(PILMask.create(get_y(im_f))))
ms[ms==1]=0
ms[ms==2]=100
print(np.unique(ms))
PILImage.create(ms)


# %%
mask = (get_yy(im_f))
np_mask = np.array(mask)
np_mask[np_mask!=2]=0
np_mask[np_mask!=1]=255
print(np.unique(np_mask))
PILImage.create(np_mask)

# %% [markdown]
# TRING A DIFFERENT WAY
# ## New Section

# %%
number_of_the_seed = 2020

random.seed(number_of_the_seed)
set_seed(number_of_the_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %%
monitor_training="valid_loss"
comp_training=np.less

monitor_evaluating="dice"
comp_evaluating=np.greater

patience=2
from albumentations import (
    Compose,
    OneOf,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    Flip,
    Rotate,
    Transpose,
    CLAHE,
    ShiftScaleRotate
)
def tumour_norm(input, target):
#input is pred (2,256,256)
    target = target.squeeze(1)
  # target is ground truth (256,256)
    if len(torch.unique(target))>5:
        mask = target !=2
    else :
        mask = target!=1
    # print(mask)
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def not_tumour(input, target):
#input is pred (2,256,256)
  target = target.squeeze(1)
  # target is ground truth (256,256)
  mask = target == void_code
  # print(mask)
  return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
def segmentron_splitter(model):
  return [params(model.backbone), params(model.head)]
## Return Jaccard index, or Intersection over Union (IoU) value
def IoU_loss(preds:Tensor, targs:Tensor, eps:float=1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value 
             for multi-class image segmentation
    """
    num_classes = preds.shape[1]
    
    # Single class segmentation?
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(preds)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        
    # Multi-class segmentation
    else:
        # Convert target to one-hot encoding
        # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
        true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]
        
        # Permute [B,H,W,C] to [B,C,H,W]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # Take softmax along class dimension; all class probs add to 1 (per pixel)
        probas = F.softmax(preds, dim=1)
        
    true_1_hot = true_1_hot.type(preds.type())
    
    # Sum probabilities by class and across batch images
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
    cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
    return 1-iou
#Return Jaccard index, or Intersection over Union (IoU) value
def IoU(preds:Tensor, targs:Tensor, eps:float=1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value 
             for multi-class image segmentation
    """
    num_classes = preds.shape[1]
    
    # Single class segmentation?
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(preds)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        
    # Multi-class segmentation
    else:
        # Convert target to one-hot encoding
        # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
        true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]
        
        # Permute [B,H,W,C] to [B,C,H,W]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # Take softmax along class dimension; all class probs add to 1 (per pixel)
        probas = F.softmax(preds, dim=1)
        
    true_1_hot = true_1_hot.type(preds.type())
    
    # Sum probabilities by class and across batch images
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
    cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
    return iou
def seg_accuracy(input:Tensor, targs:Tensor):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()


# %%
class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0
    def __init__(self, aug): 
        self.aug = aug
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])
    
transformPipeline=Compose([
                        Flip(p=0.5),
                        Transpose(p=0.5),
                        Rotate(p=0.40,limit=10)
                    ],p=1)

transformPipeline=SegmentationAlbumentationsTransform(transformPipeline)


# %%
class TargetMaskConvertTransform(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        mask = np.array(mask)
        
        li = (np.unique(mask))
        if 29 in li:
          mask[mask==29]=0
        # normal case
        if len(li)>5:
            mask[mask!=255]=0
            mask[mask==255]=2
        #tumour
        else:
            mask[mask!=255]=0
            mask[mask==255]=1
        mask = PILMask.create(mask)
        return img, mask
class TargetMaskConvertTransform_test(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        mask = np.array(mask)

       
    # Change 255 for 1
        mask[mask==1]=0
        mask[mask==2]=1

        # Back to PILMask
        mask = PILMask.create(mask)
        return img, mask

# %% [markdown]
# ### Training 

# %%
size = 256

bs = 32


def get_file_name():return original_file_names
name2id= {'NoEPI': 0, 'EPI': 1}


void_code = name2id['EPI']


# %%
class SN(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        img=np.array(img)
        #Normalise patch according to target
        img = mapping(target,img)

     
        img=PILImage.create(img)
        return img, mask


# %%
# !pip install cv
get_ipython().run_line_magic('run', 'utils.py')
import cv2 as cv

target = cv.imread(str(get_files('normal/1024/train')[2]))
target = cv.cvtColor(target, cv.COLOR_BGR2RGB)


# %%
manual = DataBlock(blocks=(ImageBlock, MaskBlock(['background','turmou','normal'])),
                   get_items=partial(get_image_files,folders=["train"]),
                   get_y=get_y_test,
                   splitter=RandomSplitter(valid_pct=0.3,seed=2020),
                   item_tfms=[Resize((size,size)),TargetMaskConvertTransform(),SN()],
                  )
# manual.summary('../input/epi-data-512-1024/512')
# dls = manual.dataloaders('../input/epi-data-512-1024/512',bs=bs,num_workers=1)

# # manual.summary('../input/epi-seg/training_PNG')
# # dls = manual.dataloaders('../input/epi-seg/training_PNG',bs=bs)


manual.summary('/content/normal/1024/')
dls = manual.dataloaders('/content/normal/1024/',bs=bs)

# manual.summary('../input/test-epi/')
# dls = manual.dataloaders('../input/test-epi/',bs=bs,num_workers=1)

dls.show_batch(vmin=0,vmax=1,figsize=(12, 9))


# %%
get_ipython().system('pip install wandb')

# !wandb login
import wandb
from fastai.callback.wandb import *

os.environ["WANDB_API_KEY"] = '4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8'


# %%
b = dls.one_batch()
get_unique(to_np(b[1][1]))


# %%

learn = get_segmentation_learner(dls=dls, number_classes=3, segmentation_type="Semantic Segmentation",
                                 architecture_name="deeplabv3+", backbone_name="resnet50", 
                                 metrics=[ seg_accuracy ,tumour_norm,Dice(), JaccardCoeff(),DiceMulti(),IoU],wd=1e-2,
                                 ).to_fp16()
learn.freeze() # Freezing the backbone


# %%
learn.loss_func=IoU_loss
str(learn.loss_func)


# %%
fname="deeplabv3-segmentron-resnet50-no-data-augmentation-before-unfreeze-WD-2-best"
CUDA_LAUNCH_BLOCKING=1
WANDB_API_KEY="4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8"
os.environ["WANDB_API_KEY"] = '4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8'

# start logging a wandb run
run=wandb.init(project='EPISEG',name=f"deeplabv3+_{learn.backbone_name}_1024down{size}_VALBoth_{str(learn.loss_func)}")


callbacksFitBeforeUnfreeze = [
    WandbCallback(),
    ShowGraphCallback(),
    EarlyStoppingCallback(monitor=monitor_training,comp=comp_training, patience=patience),
    SaveModelCallback(monitor=monitor_training,comp=comp_training,every_epoch=False,fname=fname)  
]
learn.fit_one_cycle(10, slice(3e-3,3e-2),cbs=callbacksFitBeforeUnfreeze)


# %%

learn.lr_find() # find learning rate
learn.recorder # plot learning rate graph


# %%
learn.load("deeplabv3-segmentron-resnet50-no-data-augmentation-before-unfreeze-WD-2-best")
learn.validate()


# %%
learn.unfreeze()
learn.lr_find() # find learning rate
learn.recorder # plot learning rate graph


# %%
fname="deeplabv3-segmentron-resnet50-no-data-augmentation-after-unfreeze-WD-2-best"

callbacksFitAfterUnfreeze = [
    WandbCallback(),
    ShowGraphCallback(),
    EarlyStoppingCallback(monitor=monitor_training,comp=comp_training, patience=patience),
    SaveModelCallback(monitor=monitor_training,comp=comp_training,every_epoch=False,fname=fname)  
]

learn.fit_one_cycle(10, slice(7e-7,2e-6),cbs=callbacksFitAfterUnfreeze)


# %%

learn = get_segmentation_learner(dls=dls, number_classes=3, segmentation_type="Semantic Segmentation",
                                 architecture_name="deeplabv3+", backbone_name="resnet50", 
                                 metrics=[ seg_accuracy ,tumour_norm,Dice(), JaccardCoeff(),DiceMulti(),IoU],wd=1e-1,
                                 loss_func = IoU_loss,
                                 splitter=segmentron_splitter).to_fp16()
learn.freeze() # Freezing the backbone


# %%
learn.lr_find() # find learning rate
learn.recorder # plot learning rate graph


# %%
fname="deeplabv3-segmentron-resnet50-no-data-augmentation-before-unfreeze-WD-1-best"

callbacksFitBeforeUnfreeze = [
    WandbCallback(),
    ShowGraphCallback(),
    EarlyStoppingCallback(monitor=monitor_training,comp=comp_training, patience=patience),
    SaveModelCallback(monitor=monitor_training,comp=comp_training,every_epoch=False,fname=fname)  
]
learn.fit_one_cycle(10, slice(3e-3,3e-2),cbs=callbacksFitBeforeUnfreeze)


# %%
learn.load("deeplabv3-segmentron-resnet50-no-data-augmentation-before-unfreeze-WD-1-best")
learn.validate()


# %%
learn.unfreeze()
learn.lr_find() # find learning rate
learn.recorder # plot learning rate graph


# %%
fname="deeplabv3-segmentron-resnet50-no-data-augmentation-after-unfreeze-WD-1-best"

callbacksFitAfterUnfreeze = [
#     WandbCallback(),
    ShowGraphCallback(),
    EarlyStoppingCallback(monitor=monitor_training,comp=comp_training, patience=patience),
    SaveModelCallback(monitor=monitor_training,comp=comp_training,every_epoch=False,fname=fname)  
]
learn.fit_one_cycle(10, slice(7e-7,2e-6),cbs=callbacksFitAfterUnfreeze)


# %%
test_name = "1024"

def ParentSplitter(x):
  #Split items by result of func (True for validation, False for training set).
    return Path(x).parent.name==test_name
manual = DataBlock(blocks=(ImageBlock, MaskBlock(['background','EPI'])),
                   get_items=partial(get_image_files,folders=["train","1024"]),
                   get_y=get_y,
                   splitter=FuncSplitter(ParentSplitter),
                   item_tfms=[Resize((size,size)),TargetMaskConvertTransform()],
                   batch_tfms=Normalize.from_stats(*imagenet_stats)
                  )
manual.summary('/content/')
dls = manual.dataloaders('/content/',bs=bs)
dls.show_batch(vmin=0,vmax=1,figsize=(12, 9))


# %%
learn = load_learner("./artifacts/my-model_multi:v1/export.pkl")


# %%
learn.dls=dls


# %%

learn = get_segmentation_learner(dls=dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                 architecture_name="deeplabv3+", backbone_name="resnet50", 
                                 metrics=[ Dice(), JaccardCoeff()],wd=1e-2,
                                 splitter=segmentron_splitter).to_fp16()
learn.freeze() # Freezing the backbone


# %%
learn.lr_find() # find learning rate
learn.recorder # plot learning rate graph


# %%
run=wandb.init(project='EPISEG',name =f"deeplabv3+_{learn.backbone_name}_1024down{size}_ValIHC")


# %%
learn.freeze() # Freezing the backbone


# %%
fname="deeplabv3-segmentron-resnet50-no-data-augmentation-before-unfreeze-best"

callbacksFitBeforeUnfreeze = [
    WandbCallback(),
    ShowGraphCallback(),
    EarlyStoppingCallback(monitor=monitor_evaluating,comp=comp_evaluating, patience=patience),
    SaveModelCallback(monitor=monitor_evaluating,comp=comp_evaluating,every_epoch=False,fname=fname)  
]
learn.fit_one_cycle(10, slice(3e-3,3e-2),cbs=callbacksFitBeforeUnfreeze)


# %%
learn.load("deeplabv3-segmentron-resnet50-no-data-augmentation-before-unfreeze-best")
learn.validate()


# %%
# aux=learn.model
# aux=aux.cpu()

# traced_cell=torch.jit.script(aux)
# traced_cell.save("./content/models/deeplabv3+-dataset1-data-augmentation.pth")


# %%
learn.unfreeze()
learn.lr_find() # find learning rate
learn.recorder # plot learning rate graph


# %%
fname="deeplabv3-segmentron-resnet50-data-augmentation-after-unfreeze-best"

callbacksFitAfterUnfreeze = [
    WandbCallback(),
    ShowGraphCallback(),
    EarlyStoppingCallback(monitor=monitor_evaluating,comp=comp_evaluating, patience=patience),
    SaveModelCallback(monitor=monitor_evaluating,comp=comp_evaluating,every_epoch=False,fname=fname)  
]
learn.fit_one_cycle(10, slice(1e-7,2e-7),cbs=callbacksFitAfterUnfreeze)


# %%
learn.load("deeplabv3-segmentron-resnet50-data-augmentation-after-unfreeze-best")
learn.validate()


# %%
learn.show_results(0,vmin=0,vmax=1,max_n=10,) # show results


# %%
FetchPredsCallback


# %%
learn.summary()


# %%
learn.save(file="seg-1705-01.pth")
learn.export()

# %% [markdown]
# ### Testing on external data
# **

# %%


artifact = wandb.Artifact('my-model_multi', type='model')
artifact.add_file("/content/export.pkl")
wandb.log_artifact(artifact)
run.finish() 


# %%
import wandb
run = wandb.init()
artifact = run.use_artifact('usama_ml/EPISEG/my-model_multi:v4', type='model')
artifact_dir = artifact.download()


# %%
new_learner = load_learner("artifacts/my-model_multi:v4/export.pkl")
new_learner.dls=dls


# %%
def get_unique(n):
    return np.unique(n)


# %%
new_learner.validate()


# %%
b = new_learner.dls[0].new(shuffle=True).one_batch()


# %%
b[0][7].show()


# %%
new_learner.show_results(1,vmin=0,vmax=12,max_n=100,) # show results

# %% [markdown]
# ### testing on H&E

# %%
# annotated_file_names = sorted(glob("../input/epi-seg/training_PNG/mask/*"))
# original_file_names = sorted(glob("../input/epi-seg/training_PNG/train/*"))
# len(original_file_names),len(annotated_file_names)
manual = DataBlock(blocks=(ImageBlock, MaskBlock(['background','EPI'])),
                   get_items=partial(get_image_files,folders=["train"]),
                   get_y=get_y,
                   splitter=RandomSplitter(valid_pct=0.3,seed=2020),
                   item_tfms=[Resize((1024,1024))],
                   batch_tfms=Normalize.from_stats(*imagenet_stats)
                  )
manual.summary('/content/')
dls = manual.dataloaders('/content/',bs=2)


# %%
import wandb
run = wandb.init()
artifact = run.use_artifact('usama_ml/EPISEG/my-model:v7', type='model')
artifact_dir = artifact.download()


# %%
new_learner.dls=dls


# %%
[Path(sorted(glob(f"/content/filtered/input/*"))[2])]


# %%
new_learner.predict(Path(sorted(glob("/content/filtered/input/*"))[2]))


# %%
def pred_img(i,input_images_path="/content/input"):
    test_lis = sorted(glob(f"{input_images_path}/*"))[i]
    return PILImage.create(test_lis)

def get_pred_on_test(im_index,new_learner=new_learner,input_images_path="/content/input"):
  test_dl=new_learner.dls.test_dl([Path(sorted(glob(f"{input_images_path}/*"))[im_index])])

  preds = new_learner.get_preds(dl=test_dl)

  return preds
def get_test_preds(i,bk=1,input_path=None):
  preds = get_pred_on_test(i,input_images_path=input_path)
  print(preds[0].shape)
  pred_im = (preds[0][0][bk])>.5
  return pred_im 


# %%
i=40
open_image(sorted(glob("/content/filtered/input/*"))[i])

# %% [markdown]
# Predict on input tiles 

# %%
get_ipython().system('rm /content/output/*')
get_ipython().run_line_magic('run', 'utils.py')
for i in range(len(sorted(glob("/content/filtered/input/*")))):
  print(i)
  np_to_pil(np.array(get_test_preds(i,input_path="/content/filtered/input")).astype('bool')).save(f'/content/output/{os.path.basename(sorted(glob("/content/filtered/input/*"))[i])}')


# %%
get_pred_on_tiles('/content/filtered/input','/content/output/')


# %%
def get_pred_on_tiles(input_tiles_path,output_path):
  for i,fn in enumerate((sorted(glob(f"{input_tiles_path}/*")))):
    print(f'processing : {i,fn}')
    base = os.path.basename(fn)
    dir = os.path.dirname(fn)
    pred = get_test_preds(i,input_path=input_tiles_path)
    pred_np = np.array(pred).astype('bool')
    pred_pil = np_to_pil(pred_np)
    pred_pil.save(f'{output_path}/{base}')


# %%
from mpl_toolkits.axes_grid1 import ImageGrid
num_of_test=2
fig = plt.figure(1,(100,200))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(num_of_test,2),
                 axes_pad=0.1,
                 )

x,y = -2,-1
for i in range(num_of_test):
    x+=2
    y+=2
    im_pred = pred_img(i)
    im = get_test_preds(i,new_learner)
    grid[x].imshow(im,cmap='gray',interpolation='none')
    grid[y].imshow(im_pred,cmap='gray',interpolation='none')

# %% [markdown]
# # 

# %%
sorted(glob("../input/test-epi/train/*"))


# %%
get_ipython().system('ls')

# %% [markdown]
# ### Tiling of IHC based on tissue
# This is done to run inference using a the Trained model 

# %%
open_image(test)


# %%
#generate image with bg masked out

test_im_np = open_image_np(test)

np_to_pil(apply_image_filters(test_im_np,test,save=True))


# %%
import utils
get_ipython().run_line_magic('run', 'utils.py')

test = "/content/D:\Other\DOWNLOADS\WSIData/filtered/3IFZ-02L-CD3-2012-03-2017.48.32_3794_C05R03_ImageActualTif.png"
SCALE_FACTOR = 1 
imgpath = utils.get_slide_path_based_on_PNG_path(test,test=False)
_,o_w,o_h,w,h = slide_to_scaled_pil_image(imgpath,SCALE_FACTOR)
tile_sum = score_tiles(test,test=False,SCALE_FACTOR=SCALE_FACTOR)


# %%
tile_sum.tiles_by_score


# %%
t=tile_sum.get_tile(5,4)


# %%
FILTER_DIR

# %% [markdown]
# # Generating output 

# %%
fns = sorted(glob("/content/*.png"))
#converting all images into tisuse only images (ie. bg out) 
get_ipython().run_line_magic('run', 'utils.py')
def generate_masked_image(fn):
    test_im_np = open_image_np(fn)
    return np_to_pil(apply_image_filters(test_im_np,fn,save=True))

for ind in fns[:10]:
  generate_masked_image(ind)


# %%
generate_masked_image(fns[98])


# %%
get_ipython().run_line_magic('run', 'utils.py')
#TODO: figure out a way to ameke this mroe automated 
get_ipython().system('rm /content/input/*')

def generate_input_tiles(index,path,output_dir=None):
  ind=index
  #list of filtered images.. ones without background 
  test = sorted(glob(f"{path}/*.png"))[ind]
  print(test)
  tile_sum = score_tiles(test,test=False,SCALE_FACTOR=SCALE_FACTOR)
  dir = os.path.dirname(test)
  dir = os.path.join(dir,'input')
  if  not os.path.exists(dir):
    os.makedirs(dir)
  if output_dir is None:
    output_dir=dir
  for t in tile_sum.tiles_by_score():
    c = t.c 
    r = t.r 
    ce=t.o_c_e
    rs=t.o_r_s
    re=t.o_r_e
    cs=t.o_c_s
    t.get_pil_tile().save(f"{output_dir}/{tile_sum.num_col_tiles}_{tile_sum.num_col_tiles}_{rs}_{re}_{cs}_{ce}_.png")


# %%
generate_input_tiles(10,FILTER_DIR)


# %%
test = sorted(glob(f"{FILTER_DIR}/*.png"))
test[10]

# %% [markdown]
# # Re-stitcheing output tiles 

# %%
def get_outs(out_path):
  outs= sorted(glob(f"{out_path}/*"))
  out_base = os.path.basename(outs[0])
  out_split = out_base.split("_")
  tile_num = int(out_split[0])
  return outs,tile_num
def get_tile_sum_image(tile_sum):
  fns = sorted(glob(f"{FILTER_DIR}/*.png"))

  num  = get_num(tile_sum.slide_num)
  path = [s for s in fns if num in s][0]
  return open_image(path)
def get_stitched_slide(outs,tile_num,tile_size=1024):
  re_tile = np.zeros([tile_num*tile_size, tile_num*tile_size])

  for out in outs:
    out_base = os.path.basename(out)
    print(out_base)
    out_split = out_base.split("_")
    rs=int(out_split[2])
    re=int(out_split[3])
    cs=int(out_split[4])
    ce=int(out_split[5])
    if ce-cs!=tile_size:
      ce+=(tile_size-(ce-cs))
      print(f'corrected for ce {ce}')
    if re-rs!=tile_size:
      re+=(tile_size-(re-rs))
      print(f'corrected for re {re}')
    re_tile[rs:re,cs:ce]=open_image_np(out)[:]
  return re_tile


# %%
# re stitching
outs,tile_num= get_outs("/content/output")
re_tile= get_stitched_slide(outs,tile_num)
np_to_pil(re_tile.astype('bool'))

# %% [markdown]
#  #  Jupyter Lab 

# %%
get_ipython().system('pip install kora -q')
from kora import jupyter
jupyter.start(lab=True)


# %%
jupyter.stop()


# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
get_ipython().system('eval "$SHELL"')


# %%
get_ipython().system('pip install jupyterlab')
get_ipython().system('ssh-keygen -t ed25519 -C "ososl@example.com"')
get_ipython().system('eval "$(ssh-agent -s)"')
get_ipython().system('ssh-add')


# %%
get_ipython().system('jupyter lab --ip=0.0.0.0 --port=5656 --allow-root & ssh -o StrictHostKeyChecking=no -R 80:localhost:5656 ssh.localhost.run')



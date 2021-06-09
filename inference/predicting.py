# To add a new cell, type '# %%'# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython
# from google.colab import drive
import MessageTools
import traceback
import sys
from torch.utils.data import DataLoader
# %%
from globals import *
import os 
from glob import glob
import wandb
from utils_pred import *

from pred_utils import * 
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
    
sys.path.insert(0,'/mnt/data/Other/Projects/codeServerEPI/Attention-Gated-Networks')
from dataio.transformation import get_dataset_transformation
from dataio.loader.test_dataset import TestDataset
from utils.util import json_file_to_pyobj
from  dataio.transformation.myImageTransformations  import Resize
from models import get_model
transformPipeline=Compose([
                        Flip(p=0.5),
                        Transpose(p=0.5),
                        Rotate(p=0.40,limit=10)
                    ],p=1)

transformPipeline=SegmentationAlbumentationsTransform(transformPipeline)

######################################################################################################
def generate_masked_image(fn):
  test_im_np = open_image_np(fn)
  return np_to_pil(apply_image_filters(test_im_np,fn,save=True))
def get_file_name():return original_file_names
name2id= {'NoEPI': 0, 'EPI': 1}

### FILTER ###
FILTER_DIR_names = sorted(glob(FILTER_DIR+"/*.png"))
void_code = name2id['EPI']
if not os.path.exists('filtered/') or len(FILTER_DIR_names)<NUM_FILTERED:
  MessageTools.show_yellow("Generating filtered slides ....")
  # # Generating output 
  fns = sorted(glob(SLIDES_PATH+'/*.png'))
  MessageTools.show_blue("converting all images into tisuse only images (ie. bg out)")

  for ind in fns[:NUM_FILTERED]:
    generate_masked_image(ind)

else:
  MessageTools.show_blue("Filtered dir already exists. Continue ..")

if PROCESS_SLIDE is not 'any':
  if not sum([False if PROCESS_SLIDE not in x else True for x in FILTER_DIR_names ]):
    MessageTools.show_yellow("Slide not in filtered .. trying to find it")
    try:
      path = [p for p in glob(SLIDES_PATH +'/*') if PROCESS_SLIDE in p][0]
      generate_masked_image(path)
    except Exception as e:
      MessageTools.show_err(f"Can't find slide! --> {e}")


# %%
MessageTools.show_yellow("Assigining Target")
# !pip install cv
import cv2 as cv

target = cv.imread('inference/test.png')
target = cv.cvtColor(target, cv.COLOR_BGR2RGB)





os.environ["WANDB_API_KEY"] = '4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8'


# %%
MessageTools.show_yellow(f"Downlaoding model")
if os.path.exists(f'artifacts/attention_model_unet:v{MODEl_VERSION}/'):
  MessageTools.show_blue("Model already exists")
  artifact_dir = os.path.join(f'artifacts/attention_model_unet:v{MODEl_VERSION}','')
else:
  import wandb
  run = wandb.init(resume=True)
  artifact = run.use_artifact(f'usama_ml/EPISEG/attention_model_unet:v{MODEl_VERSION}', type='model')
  artifact_dir = artifact.download()
model_path = os.path.join(artifact_dir,os.listdir(artifact_dir)[0])

######################################################################################################
MessageTools.show_yellow(f"Generating input for slide")
if RECURSE ==True and PROCESS_SLIDE=='any':
  global slide_num
  pass
else:
  NUM_FILTERED=1
for i in range(NUM_FILTERED):
    os.system('rm filtered/input/*')

    def generate_input_tiles(index,path,output_dir=None):
      ind=index
      #list of filtered images.. ones without background 
      test = sorted(glob(path+"/*.png"))
      test.extend(sorted(glob(path+"/*.tif")))
      test=test[ind]
      MessageTools.show_succ(test)
      tile_sum = score_tiles(test,test=True,SCALE_FACTOR=SCALE_FACTOR)
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
        tily=t.get_pil_tile()
        tily=np.array(tily)
        if VERBOSE > 0:
          print(rs,re,cs,ce,tily.shape)
        try:
          mapped=mapping(target,tily)
        except Exception as e:
          if VERBOSE > 0:
            MessageTools.show_err("Can't normalise tile using as is !")
          mapped=tily
        tily_m=Image.fromarray(mapped)
        tily_m.save(f"{output_dir}/{tile_sum.num_row_tiles}_{tile_sum.num_col_tiles}_{rs}_{re}_{cs}_{ce}_.png")

    slide_num=False
    def get_slide_idx(name,idx=None):
      if name =="any":
        len_filtered=len(FILTER_DIR_names)
        #randomly go thorugh filtered images
        if idx is None:
          MessageTools.show_yellow("Randomly choose a filtered Image")
          random_idx=random.randint(0, len_filtered-1)	
        else:
          MessageTools.show_yellow("Iterating through filtered images")
          random_idx=idx
        if VERBOSE > 0:
          print(random_idx,len_filtered)
          #get slide num for file saving 
        slide_num=get_num_norm(FILTER_DIR_names[random_idx])
        MessageTools.show_blue(f"Going for slide {slide_num} at idx {random_idx} in filtered")
        return random_idx,slide_num
        
      for idx,val in enumerate(sorted(glob(FILTER_DIR+'/*'))):
        if name in val:
          return idx,name
    # %%
    try:
      arg={'name':PROCESS_SLIDE,'idx':None if RANDOM_RUN else i}
      slide_idx,slide_num=get_slide_idx(**arg)
      generate_input_tiles(slide_idx,FILTER_DIR)
    except Exception as e:
      traceback.print_exc()
      MessageTools.show_err(f"Failed to generate input --> {e}")


    # Predict on input tiles 
    ######################################################################################################

# %% creating test data
    json_filename = "configs/config_TransUnet_test.json"
    json_opts = json_file_to_pyobj(json_filename)
    ds_transform = get_dataset_transformation('test', opts=json_opts.augmentation)
    dls = TestDataset('filtered/input',transform=ds_transform['train'])
    batchSize=1
    test_loader  = DataLoader(dataset=dls,  num_workers=0, batch_size=batchSize, shuffle=False)


    model = get_model(json_opts.model,model_path = model_path)
    # %%
    MessageTools.show_yellow("Running predictions ...")
    os.system('rm output/*')
    if not os.path.exists('output/'):
      os.makedirs('output/')
    for iteration, (images,f_paths, labels) in enumerate(test_loader, 1):

      model.set_input(images)
      model.test()

      out = model.logits
      out = out.cpu().argmax(dim=1).squeeze().numpy().astype(np.uint8)
      RGB = np.zeros((out.shape[0],out.shape[1],3), dtype=np.uint8)
      #rgb convert for better res visulisation
      RGB[out==0] = [200,0,0] # bg is red
      RGB[out==1] = [0,0,255] # tumour is blue
      RGB[out==2] = [0,255,0] # normal is green
      pred_pil= np_to_pil(RGB)
      # out = np_to_pil(out.squeeze() * 255)
      out = pred_pil
      out = out.resize((1024,1024),PIL.Image.LANCZOS)
      out.save(f'output/{os.path.basename(f_paths[0])}')




    ######################################################################################################
    # %%
    MessageTools.show_succ("Preds Done .. stitching")
    # %%
    def get_outs(out_path):
      if not os.path.exists(out_path):
        os.makedirs(out_path)
      outs= sorted(glob(f"{out_path}/*"))
      out_base = os.path.basename(outs[0])
      out_split = out_base.split("_")
      num_rows= int(out_split[0])
      num_cols= int(out_split[1])
      return outs,num_rows,num_cols
    def get_tile_sum_image(tile_sum):
      fns = sorted(glob(f"{FILTER_DIR}/*.png"))

      num  = get_num(tile_sum.slide_num)
      path = [s for s in fns if num in s][0]
      return open_image(path)
    def get_stitched_slide(outs,num_rows,num_cols,tile_size=1024):
      re_tile = np.zeros((num_rows*tile_size,num_cols*tile_size,3))

      for out in outs:
        out_base = os.path.basename(out)
        if VERBOSE > 0:
          print(out_base)
        out_split = out_base.split("_")
        rs=int(out_split[2])
        re=int(out_split[3])
        cs=int(out_split[4])
        ce=int(out_split[5])
        if ce-cs!=tile_size:
          ce+=(tile_size-(ce-cs))
          if VERBOSE > 0:
            print(f'corrected for ce {ce}')
        if re-rs!=tile_size:
          re+=(tile_size-(re-rs))
          if VERBOSE > 0:
            print(f'corrected for re {re}')
        re_tile[rs:re,cs:ce,:]=open_image_np(out)[:]
      return re_tile


    # %%
    # re stitching
    if slide_num == False: 
      slide_num=PROCESS_SLIDE
    if SAVE_DRIVE:
      extra_path='drive/My Drive/stitched/'
    else:
      extra_path=''
    outs,num_rows,num_cols= get_outs("output")
    re_tile= get_stitched_slide(outs,num_rows,num_cols)
    MessageTools.show_yellow(f"saving to ---> output_slides/{extra_path}{slide_num}_stitched_model_{MODEl_VERSION}.png")
    if not os.path.exists('output_slides/'):
      os.makedirs('output_slides/')
    np_to_pil(re_tile.astype('bool')).save(f"output_slides/{extra_path}{slide_num}_stitched_model_{MODEl_VERSION}.png")



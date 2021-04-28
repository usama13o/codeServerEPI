# To add a new cell, type '# %%'# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython
from google.colab import drive
import MessageTools
import traceback
# %%
from globals import *
import os 
from semtorch import get_segmentation_learner
from glob import glob
from fastai import *
from fastai.vision.all import *
from fastai.vision import *

import wandb
from fastai.callback.wandb import *
from utils import *
from pred_utils import * 

MessageTools.show_succ(f"Using GLOBALS: \n SLIDES_PATH: {SLIDES_PATH} |NUM_FILTERED:  {NUM_FILTERED} | TARG_PRED: {TARG_PRED} [{NORMAL if TARG_PRED ==2 else TUMOUR}] | PROCESS_SLIDE: {PROCESS_SLIDE} |")
if SAVE_DRIVE:
  drive.mount('/content/drive')
if os.path.exists('/content/normal-tifff') or os.path.exists('/content/epithelial-cells-ihc'):
  MessageTools.show_blue("Data already Unzipped !")
  pass
else:
  MessageTools.show_yellow("Unzipping data ...")

# %%
  for zfile in glob('/content/*.zip'):
    os.system(f"unzip -q {zfile} -d /content/{os.path.splitext(zfile.split('/')[-1])[0]}")
    os.system(f"rm {zfile}")
  # os.system('unzip -q /content/normal-epi -d /content/normal-epi')
  # os.system('rm  /content/normal-epi.zip')

  # os.system('unzip -q /content/epithelial-cells-ihc -d /content/ihc')
  # os.system('rm  /content/epithelial-cells-ihc.zip')

# %%
  # os.system('unzip -q /content/normal-tifff -d /content/normal')
  # os.system('rm /content/normal-tifff.zip')
try:
  len_ihc = len([name for name in os.listdir('/content/normal-epi/') ])
  #TODO Figure out something here !
  len_norm = len([name for name in os.listdir('/content/normal/')])
  MessageTools.show_yellow(f'Data all here ! length-_->  normal:  {len_norm} normal-epi:  {len_ihc}')
except Exception as e :
  MessageTools.show_err(f"Something is missing, Failed to show file stats --> {e}")

# %%
# %% [markdown]
# TRING A DIFFERENT WAY
# ## New Section

# %%
# number_of_the_seed = 2020

# random.seed(number_of_the_seed)
# set_seed(number_of_the_seed)

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
    
transformPipeline=Compose([
                        Flip(p=0.5),
                        Transpose(p=0.5),
                        Rotate(p=0.40,limit=10)
                    ],p=1)

transformPipeline=SegmentationAlbumentationsTransform(transformPipeline)


# %%

# %% [markdown]
# ### Training 

# %%
size = 256

bs = 32


######################################################################################################
def generate_masked_image(fn):
  test_im_np = open_image_np(fn)
  return np_to_pil(apply_image_filters(test_im_np,fn,save=True))
def get_file_name():return original_file_names
name2id= {'NoEPI': 0, 'EPI': 1}

### FILTER ###
void_code = name2id['EPI']
if not os.path.exists('/content/filtered/') or len(FILTER_DIR_names)<NUM_FILTERED:
  MessageTools.show_yellow("Generating filtered slides ....")
  # # Generating output 
  fns = sorted(glob(SLIDES_PATH+'/*'))
  MessageTools.show_blue("converting all images into tisuse only images (ie. bg out)")

  for ind in fns[:NUM_FILTERED]:
    generate_masked_image(ind)

else:
  MessageTools.show_blue("Filtered dir already exists. Continue ..")

if PROCESS_SLIDE is not 'any':
  if not [False for x in FILTER_DIR_names if PROCESS_SLIDE in x][0]:
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

target = cv.imread('codeServerEPI/test.png')
target = cv.cvtColor(target, cv.COLOR_BGR2RGB)



# %%


os.environ["WANDB_API_KEY"] = '4d3d06d5a500f0245b15ee14cc3b784a37e2d7e8'


# %%

# %%
# MessageTools.show_yellow('Making learner ...')
# learn = get_segmentation_learner(dls=dls, number_classes=3, segmentation_type="Semantic Segmentation",
                                #  architecture_name="deeplabv3+", backbone_name="resnet50", 
                                #  metrics=[ seg_accuracy ,tumour_norm,Dice(), JaccardCoeff(),DiceMulti(),IoU],wd=1e-2,
                                #  ).to_fp16()

CUDA_LAUNCH_BLOCKING=1

######################################################################################################

def get_y_fake(im_f):
  return im_f
# %%
import wandb

if not os.path.exists('/content/artifacts/'):
  MessageTools.show_yellow("Gettiing model .. Downloading ... ")
  run = wandb.init()
  artifact = run.use_artifact('usama_ml/EPISEG/my-model_multi:v4', type='model')
  artifact_dir = artifact.download()
else:
  MessageTools.show_blue('Model already downloaded at /artifacts/')
MessageTools.show_yellow("Making DataBlock .. here we go !")
try:
  manual = DataBlock(blocks=(ImageBlock, MaskBlock(['background','Tumour','Normal'])),
                    get_items=partial(get_image_files,folders=["train"]),
                    get_y= get_y_fake,
                    splitter=RandomSplitter(valid_pct=0.3,seed=2020),
                    item_tfms=[Resize((1024,1024))],
                    )
  dls = manual.dataloaders('/content/epithelial-cells-ihc',bs=2)

  new_learner = load_learner("artifacts/my-model_multi:v4/export.pkl")
  new_learner.dls=dls
  MessageTools.show_succ("Model and dls Loaded successfully")
except Exception as e:
  MessageTools.show_err(f"Falied to load DataBlock ---> {e}")
# %%
def get_unique(n):
    return np.unique(n)

######################################################################################################
MessageTools.show_yellow(f"Generating input for slide")
if RECURSE ==True and PROCESS_SLIDE=='any':
  global slide_num
  pass
else:
  NUM_FILTERED=1
for i in range(NUM_FILTERED):
    os.system('rm /content/filtered/input/*')

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
    # %% how to predict ona  single image 
    # new_learner.predict(Path(sorted(glob("/content/filtered/input/*"))[2]))


    # %% [markdown]
    # Predict on input tiles 
    ######################################################################################################

    # %%
    MessageTools.show_yellow("Running predictions ...")
    os.system('rm /content/output/*')
    if not os.path.exists('/content/output/'):
      os.makedirs('/content/output/')
    # for i in range(len(sorted(glob("/content/filtered/input/*")))):
    #   if VERBOSE > 0:
    #     print(i)
    #   np_to_pil(np.array(get_test_preds(i,learner=new_learner,bk=TARG_PRED,input_path="/content/filtered/input")).astype('bool')).save(f'/content/output/{os.path.basename(sorted(glob("/content/filtered/input/*"))[i])}')
    get_test_preds_batch(learner=new_learner,prob=0.9,bk=TARG_PRED,input_path="/content/filtered/input",output_path="/content/output")
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
    outs,num_rows,num_cols= get_outs("/content/output")
    re_tile= get_stitched_slide(outs,num_rows,num_cols)
    MessageTools.show_yellow(f"saving to ---> /content/{extra_path}{slide_num}_stitched_{TARG_PRED}.png")
    np_to_pil(re_tile.astype('bool')).save(f"/content/{extra_path}{slide_num}_stitched_{TARG_PRED}.png")



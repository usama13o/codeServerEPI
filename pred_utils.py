from pathlib import Path
import os
from glob import glob 
import numpy as np
from utils import np_to_pil
if __name__ == "__main__":
   print("Pred utils executed when ran directly")
else:
   print("Pred utils imported")
# %%
def get_pred_on_test(im_index,new_learner,input_images_path="/content/input"):
  test_dl=new_learner.dls.test_dl([Path(sorted(glob(f"{input_images_path}/*"))[im_index])])
  preds = new_learner.get_preds(dl=test_dl)

  return preds
def get_batch_preds(new_learner,input_images_path="/content/input"):
  data = sorted(glob(input_images_path+'/*'))
  data=list(map(Path,data))
  test_dl=new_learner.dls.test_dl(data)
  preds = new_learner.get_preds(dl=test_dl)
  return preds[0],test_dl.items
def get_test_preds_batch(learner,bk=1,prob=.5,input_path=None,output_path='/content/output/'):
  preds,fns = get_batch_preds(learner,input_images_path=input_path)
  for i,pred in enumerate(preds):
    print(f'processing : {i,fns[i]}')
    base= os.path.basename(str(fns[i]))
    pred=(pred[bk])>prob
    pred_np=np.array(pred).astype('bool')
    pred_pil= np_to_pil(pred_np)
    pred_pil.save(f'{output_path}/{base}')
  
def get_test_preds(i,learner,bk=1,input_path=None,output_path='/content/output/'):
  preds = get_pred_on_test(i,new_learner=learner,input_images_path=input_path)
  print(preds[0].shape)
  pred_im = (preds[0][0][bk])>.5
  return pred_im 
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


from pathlib import Path
from glob import glob 
if __name__ == "__main__":
   print("Pred utils executed when ran directly")
else:
   print("Pred utils imported")
# %%
def pred_img(i,input_images_path="/content/input"):
    test_lis = sorted(glob(f"{input_images_path}/*"))[i]
    return PILImage.create(test_lis)

def get_pred_on_test(im_index,new_learner,input_images_path="/content/input"):
  test_dl=new_learner.dls.test_dl([Path(sorted(glob(f"{input_images_path}/*"))[im_index])])

  preds = new_learner.get_preds(dl=test_dl)

  return preds
def get_test_preds(i,learner,bk=1,input_path=None):
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


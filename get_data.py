import os
import subprocess
import sys
os.environ['KAGGLE_CONFIG_DIR'] = "/content/"

get_ipython().system('apt-get install openslide-tools')
get_ipython().system('pip install openslide-python')
get_ipython().system('pip install scipy')
get_ipython().system('pip install fastai --upgrade')
get_ipython().system('pip install semtorch')
get_ipython().system('pip install spams')
get_ipython().system('pip install wandb')


list_datasets=[
"kaggle datasets download -d usamann/epithelial-cells-ihc",
"kaggle datasets download -d usamann/normal-tifff"
]

for link in list_datasets:
    subprocess.check_call(link.split())
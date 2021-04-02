import os
import subprocess
import sys
os.environ['KAGGLE_CONFIG_DIR'] = "/content/codeServerEPI/"

os.system('apt-get install openslide-tools')
os.system('pip install openslide-python')
os.system('pip install scipy')
os.system('pip install fastai --upgrade')
os.system('pip install semtorch')
os.system('pip install spams')
os.system('pip install wandb')


list_datasets=[
"kaggle datasets download -d usamann/epithelial-cells-ihc",
"kaggle datasets download -d usamann/normal-tifff"
]

for link in list_datasets:
    subprocess.check_call(link.split())
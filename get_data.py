import os

import subprocess
import sys
os.environ['KAGGLE_CONFIG_DIR'] = "/content/codeServerEPI/"

os.system('pip install scipy')
os.system('pip install semtorch')
os.system('pip install spams')
os.system('pip install wandb')
os.system('pip install pywick')
os.system('git config --global user.email "osama.zadan@gmail.com"')
os.system('git config --global user.name "usama13o"')


list_datasets=[
"kaggle datasets download -d usamann/stain-data",
# "kaggle datasets download -d usamann/epithelial-cells-ihc",
# "kaggle datasets download -d usamann/normal-epi",
# "kaggle datasets download -d usamann/normal-tifff"
]

for link in list_datasets:
    subprocess.check_call(link.split())
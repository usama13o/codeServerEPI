import os

import subprocess
import sys
<<<<<<< HEAD
os.environ['KAGGLE_CONFIG_DIR'] = "./"

=======
os.environ['KAGGLE_CONFIG_DIR'] = "/content/codeServerEPI/"

os.system('apt-get install openslide-tools')
os.system('pip install openslide-python')
os.system('pip install scipy')
os.system('pip install fastai --upgrade')
os.system('pip install semtorch')
os.system('pip install spams')
os.system('pip install wandb')
>>>>>>> master
os.system('git config --global user.email "osama.zadan@gmail.com"')
os.system('git config --global user.name "usama13o"')


list_datasets=[
<<<<<<< HEAD
"kaggle datasets download -d usamann/staindata",
# "kaggle datasets download -d usamann/epithelial-cells-ihc",
=======
"kaggle datasets download -d usamann/epithelial-cells-ihc",
>>>>>>> master
# "kaggle datasets download -d usamann/normal-epi",
# "kaggle datasets download -d usamann/normal-tifff"
]

for link in list_datasets:
<<<<<<< HEAD
    subprocess.check_call(link.split())
os.system('unzip staindata.zip')
os.system('pip install -r requriments.txt')
# os.system('rm staindata.zip')
os.system('python pywich_test.py')
=======
    subprocess.check_call(link.split())
>>>>>>> master

import os

import subprocess
import sys
os.environ['KAGGLE_CONFIG_DIR'] = "./"
os.system('pip install -r requriments.txt')

os.system('git config --global user.email "osama.zadan@gmail.com"')
os.system('git config --global user.name "usama13o"')

list_datasets=[
"kaggle datasets download -d usamann/staindata",

"kaggle datasets download -d usamann/epithelial-cells-ihc",
"kaggle datasets download -d usamann/normal-epi",
"kaggle datasets download -d usamann/normal-tifff"

]

for link in list_datasets:
    subprocess.check_call(link.split())

os.system('tar -xf staindata.zip')
os.system('tar -xf epithelial-ihc.zip')
os.system('tar -xf normal.zip')
os.system('rm staindata.zip')
os.system('python3 pywich_test.py')

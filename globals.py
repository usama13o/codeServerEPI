####GLOBALS####
NORMAL='NORMAL'
TUMOUR='TUMOUR'
#Model version 
MODEL_VER=5
MODEL_BASE='my-model_multi'
#num of filtered slides to generate 
NUM_FILTERED=10
#which slide of the filtered to process
PROCESS_SLIDE='any'
#where are the slides to be filtered. format --> '/PATH/to/slides/' 
# SLIDES_PATH='/content/normal-tifff'
# DATALOADER_PATH= 'normal-epi/1024'

SLIDES_PATH='/content/epithelial-cells-ihc'
DATALOADER_PATH= 'epithelial-cells-ihc'
#pred for tumour {1} or normal {2} cells
TARG_PRED=1
# verbosity
VERBOSE=1
#save to google drive
SAVE_DRIVE=True
#recurse through all filtered images | run on one only
RECURSE=True
#randomly choose idx of filtered if not then iterate thorugh all!
RANDOM_RUN = False
####GLOBALS####
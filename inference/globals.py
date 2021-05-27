####GLOBALS####
NORMAL='NORMAL'
TUMOUR='TUMOUR'
#num of filtered slides to generate 
NUM_FILTERED=6
#which slide of the filtered to process
PROCESS_SLIDE='3794'
#where are the slides to be filtered. format --> '/PATH/to/slides' 
SLIDES_PATH='/mnt/data/Other/DOWNLOADS/WSIData/training_PNG'
# SLIDES_PATH='/mnt/data/Other/DOWNLOADS/WSIData/Normal/Original/PNG'
#pred for tumour {1} or normal {2} cells
TARG_PRED=2
# verbosity
VERBOSE=1
#save to google drive
SAVE_DRIVE=False
#recurse through all filtered images | run on one only
RECURSE=True
#randomly choose idx of filtered if not then iterate thorugh all!
RANDOM_RUN =False
#model version
MODEl_VERSION = '215'
####GLOBALS####
if SLIDES_PATH != 'any':
    RECURSE = False
    RANDOM_RUN = False
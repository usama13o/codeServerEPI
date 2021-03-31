import torch 
import pandas
import globals

import MessageTools
# from scipy.misc import imread
import math
import matplotlib.pyplot as plt
import multiprocessing
import PIL
from PIL import Image
from fastai import *
from fastai.vision.all import *
from fastai.vision import *
import openslide
import os
from glob import glob
import sys
import numpy as np
import utils
from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
import scipy.ndimage.morphology as sc_morph
from enum import Enum
#Consts file names
DEST_MAIN_DIR="/content"
BASE_PNG_DIR="training_PNG"
FILTER_DIR =os.path.join(DEST_MAIN_DIR,"filtered")
PNG_org_DIR = os.path.join(DEST_MAIN_DIR,BASE_PNG_DIR)
PNG_ANN_DIR = os.path.join(DEST_MAIN_DIR,BASE_PNG_DIR+"_annotation")
annotated_file_names = sorted(glob("/content/1024_mask/1024/*"))
original_file_names= sorted(glob("/content/D:\Other\DOWNLOADS\WSIData/filtered/*"))
annotated_test_dir = sorted( glob( FILTER_DIR + '/*'))
annotated_tif_test_dir = sorted(glob("D:\Other\DOWNLOADS\WSIData\\filtered\\annotated_tif\*"))
PNG_original_file_names = sorted(glob(PNG_org_DIR+"\*"))
PNG_annotated_file_names =sorted(glob(PNG_ANN_DIR+"\*"))
FILTER_DIR_names = sorted(glob(FILTER_DIR+"/*.png"))
FILTER_DIR_names.extend(sorted(glob(FILTER_DIR+"/*.tif")))
NORM_TIFF_DIR = sorted(glob(f'{globals.SLIDES_PATH}*'))
SCALE_FACTOR = 1
TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10
ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
NUM_TOP_TILES = 200 
DISPLAY_TILE_SUMMARY_LABELS = True
TILE_LABEL_TEXT_SIZE = 5
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = True
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = True 
TILE_BORDER_SIZE = 4  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

FONT_PATH = "D:/Other/DOWNLOADS/ae_Salem.ttf"
SUMMARY_TITLE_FONT_PATH = "D:/Other/DOWNLOADS/ae_Salem.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330



# !pip install --no-deps openslide-python
# !apt-get install openslide-tools
import openslide
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
import torch 
import pandas
# !pip install scipy==1.1.0
import math
import matplotlib.pyplot as plt
import multiprocessing
import PIL
from PIL import Image
import openslide
import os
from glob import glob
import sys
import numpy as np

from tqdm.auto import tqdm

def get_num_lis(lis):
    lis = [x.split("_")[1] for x in lis if len(x.split("_")) >1 ]
    return lis
def get_num_norm(path):
  base = os.path.basename(path)
  fn = os.path.splitext(base)[0]
  return fn
def get_num(path):# print(os.path.basename(path).split("_"))
    if (len(os.path.basename(path).split("_"))==1):
      MessageTools.show_yellow("using the normal path num retrival")
      return get_num_norm(path)
    return os.path.basename(path).split("_")[1]


def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image
def open_slide(filename):
    """
    Open a whole-slide image (*.svs, etc).
    Args:
    filename: Name of the slide file.
    Returns:
    An OpenSlide object representing a whole-slide image.
    """
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide
def slide_to_scaled_pil_image(path,scale_factor=1):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    Args:
    slide_number: The slide number.
    Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    SCALE_FACTOR = scale_factor
    MessageTools.show_yellow("Scaling to %d ..."%(SCALE_FACTOR))
    
    slide_filepath = path
    MessageTools.show_yellow("Opening Slide : %s" % ( slide_filepath))
    slide = open_slide(slide_filepath)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h
def training_slide_to_image(path,DEST_TRAIN_DI,scale_factor=1,test=False):
    """
    Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.
    Args:
    slide_number: The slide number.
    """
    if (DEST_TRAIN_DI ==0):
      DEST_TRAIN_DI = os.path.join( "D:\Other\DOWNLOADS\WSIData","training_PNG")
    
    MessageTools.show_yellow("Scaling to %d ..."%(scale_factor))
    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(path,scale_factor)
# for original [35:-4] , annotated [37:-4]
    path = get_num_norm(path)
    img_path = os.path.join(DEST_TRAIN_DI,path+".png")
    MessageTools.show_yellow("Saving image to: " +img_path)
    if not os.path.exists(DEST_TRAIN_DIR):
        os.makedirs(DEST_TRAIN_DIR)
    img.save(img_path)

    #     thumbnail_path = get_training_thumbnail_path(slide_number, large_w, large_h, new_w, new_h)
    #     save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)        
def test_slide_to_image(path,scale_factor = 1):
  
  dest  = os.path.dirname(path)
  MessageTools.show_yellow("scalling to %d .. "%(scale_factor))
  img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(path,scale_factor)
  img_path =(dest+"\\PNG\\_"+get_num(path)+"_.png")
  if not os.path.exists(os.path.dirname(img_path)) :
    os.makedirs(os.path.dirname(img_path))
  MessageTools.show_yellow("Saving image to: " +img_path)
  img.save(img_path)

def slide_info(path,display_all_properties=False):
    """
    Display information (such as properties) about training images.
    Args:
    display_all_properties: If True, display all available slide properties.
    """
    slide_num=1
    num_train_images = 1
    obj_pow_20_list = []
    obj_pow_40_list = []
    obj_pow_other_list = []
    for slide_num in range(1, num_train_images + 1):
        slide_filepath = (path)
        print("\nOpening Slide #%d: %s" % (slide_num, path))
        slide = open_slide(path)
        print("Level count: %d" % slide.level_count)
        print("Level dimensions: " + str(slide.level_dimensions))
        print("Level downsamples: " + str(slide.level_downsamples))
        print("Dimensions: " + str(slide.dimensions))
        print("Associated images:")
        for ai_key in slide.associated_images.keys():
            print("  " + str(ai_key) + ": " + str(slide.associated_images.get(ai_key)))
        print("Format: " + str(slide.detect_format(path)))
        if display_all_properties:
            print("Properties:") 
            for prop_key in slide.properties.keys():
                print("  Property: " + str(prop_key) + ", value: " + str(slide.properties.get(prop_key)))

def slide_stats(_dir):
    if "annotated" not in _dir:
        target = "stats_orginal"
    else:
        target = "Stats_annotated"
        
    STATS_DIR = os.path.join(DEST_MAIN_DIR,target)
    """
    Display statistics/graphs about training slides.
    """

    if not os.path.exists(STATS_DIR):
        os.makedirs(STATS_DIR)

    num_train_images = len(_dir)
    slide_stats = []
    for idx,slide_num in enumerate(_dir):
        slide_filepath = (slide_num)
        print("Opening Slide #%d: %s" % (idx,slide_filepath))
        slide = open_slide(slide_filepath)
        (width, height) = slide.dimensions
        print("  Dimensions: {:,d} x {:,d}".format(width, height))
        slide_stats.append((width, height))

    max_width = 0
    max_height = 0
    min_width = sys.maxsize
    min_height = sys.maxsize
    total_width = 0
    total_height = 0
    total_size = 0
    which_max_width = 0
    which_max_height = 0
    which_min_width = 0
    which_min_height = 0
    max_size = 0
    min_size = sys.maxsize
    which_max_size = 0
    which_min_size = 0
    for z in range(0, num_train_images):
        (width, height) = slide_stats[z]
        if width > max_width:
            max_width = width
            which_max_width = z + 1
        if width < min_width:
            min_width = width
            which_min_width = z + 1
        if height > max_height:
            max_height = height
            which_max_height = z + 1
        if height < min_height:
            min_height = height
            which_min_height = z + 1
        size = width * height
        if size > max_size:
            max_size = size
            which_max_size = z + 1
        if size < min_size:
            min_size = size
            which_min_size = z + 1
        total_width = total_width + width
        total_height = total_height + height
        total_size = total_size + size

    avg_width = total_width / num_train_images
    avg_height = total_height / num_train_images
    avg_size = total_size / num_train_images

    stats_string = ""
    stats_string += "%-11s {:14,d} pixels (slide #%d)".format(max_width) % ("Max width:", which_max_width)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_height) % ("Max height:", which_max_height)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_size) % ("Max size:", which_max_size)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_width) % ("Min width:", which_min_width)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_height) % ("Min height:", which_min_height)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_size) % ("Min size:", which_min_size)
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_width)) % "Avg width:"
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_height)) % "Avg height:"
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_size)) % "Avg size:"
    stats_string += "\n"
    print(stats_string)

    stats_string += "\nslide number,width,height"
    for i in range(0, len(slide_stats)):
        (width, height) = slide_stats[i]
        stats_string += "\n%d,%d,%d" % (i + 1, width, height)
    stats_string += "\n"

    stats_file = open(os.path.join(STATS_DIR, "stats.txt"), "w")
    stats_file.write(stats_string)
    stats_file.close()

#   t.elapsed_display()

    x, y = zip(*slide_stats)
    colors = np.random.rand(num_train_images)
    sizes = [10 for n in range(num_train_images)]
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.title("SVS Image Sizes")
    plt.set_cmap("prism")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "svs-image-sizes.png"))
    plt.show()

    plt.clf()
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.title("SVS Image Sizes (Labeled with slide numbers)")
    plt.set_cmap("prism")
    for i in range(num_train_images):
        snum = i + 1
        plt.annotate(str(snum), (x[i], y[i]))
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "svs-image-sizes-slide-numbers.png"))
    plt.show()

    plt.clf()
    area = [w * h / 1000000 for (w, h) in slide_stats]
    plt.hist(area, bins=64)
    plt.xlabel("width x height (M of pixels)")
    plt.ylabel("# images")
    plt.title("Distribution of image sizes in millions of pixels")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "distribution-of-svs-image-sizes.png"))
    plt.show()

    plt.clf()
    whratio = [w / h for (w, h) in slide_stats]
    plt.hist(whratio, bins=64)
    plt.xlabel("width to height ratio")
    plt.ylabel("# images")
    plt.title("Image shapes (width to height)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "w-to-h.png"))
    plt.show()

    plt.clf()
    hwratio = [h / w for (w, h) in slide_stats]
    plt.hist(hwratio, bins=64)
    plt.xlabel("height to width ratio")
    plt.ylabel("# images")
    plt.title("Image shapes (height to width)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "h-to-w.png"))
    plt.show()
    
def to_np(img):
    return np.asanyarray(img)

def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.

    Shape (h, w, c) to (h, w).

    Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

    Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    return grayscale
def display_img(np_img, text=None, font_path="D:/Other/DOWNLOADS/ae_Salem.ttf", size=48, color=(255, 0, 0), background=(255, 255, 255), border=(0, 0, 0), bg=False):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()
def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
    np_img: The image represented as a NumPy array.
    Returns:
    The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)
def mask_epi(img):
    return img==0
def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800):
    """
    Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
    colored based on the average color for that segment.
    Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
    """
    labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments, multichannel=True)
    result = sk_color.label2rgb(labels, np_img, kind='avg',bg_label=0)
    return result
def filter_rag_threshold(np_img, compactness=10, n_segments=800, threshold=9):
    """
    Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
    similar regions based on threshold value, and then output these resulting region segments.
    Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    threshold: Threshold value for combining regions.
    Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
    """
    labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments, multichannel=True)
    g = sk_future.graph.rag_mean_color(np_img, labels)
    labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
    result = sk_color.label2rgb(labels2, np_img, kind='avg',bg_label=0)
    return result
def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
    """
    Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
    Closing can be used to remove small holes.
    Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for closing.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).
    Returns:
    NumPy array (bool, float, or uint8) following binary closing.
    """
    if np_img.dtype == "uint8":
        np_img = np_img / 255
    result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    if((np.unique(array))[0]==0):
      array[array!=255]=0
    return array
def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
    pil_img: The PIL Image.
    Returns:
    The PIL image converted to a NumPy array.
    """
    rgb = np.asarray(pil_img)
    return rgb
def get_tile_train_tile(tile):
    #input is tile annotation image as np
    #output is train mask where epithilial tissue is 1 else is 0 
    
    trainT =  (~tile)==255
    return trainT

def small_to_large_mapping(small_pixel, large_dimensions):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
    small_pixel: The scaled-down width and height.
    large_dimensions: The width and height of the original whole-slide image.
    Returns:
    Tuple consisting of the scaled-up width and height.
    """
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y
def summary_and_tiles(path, display=True, save_summary=False, save_data=True, save_top_tiles=True):
    """
    Generate tile summary and top tiles for slide.
    Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    """
    img_path = path
    np_img = to_np(open_image(img_path))

    tile_sum = score_tiles(slide_num, np_img)
    if save_data:
        save_tile_data(tile_sum)
    generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    if save_top_tiles:
        for tile in tile_sum.top_tiles():
            tile.save_tile()
    return tile_sum
def parse_dimensions_from_image_filename(filename):
    small_w,small_h,_ = to_np(open_image(filename)).shape
    
    return small_w, small_h,small_w, small_h

def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
    a column tile size.
    Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.
    Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
    """
    num_row_tiles = math.ceil(rows / row_tile_size)
    num_col_tiles = math.ceil(cols / col_tile_size)
    return num_row_tiles, num_col_tiles
def get_mask_focused(ann_path):
#return np array , this opens the anotation PNG and converts it to a binary mask 
    return (( ((open_image_np(get_annotation_path(ann_path))==255)[:,:,0]+  (open_image_np(get_annotation_path(ann_path))==0)[:,:,0] ) ))
def mask_percent(np_img):
    """
    How many zeros 
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
    np_img: Image as a NumPy array.
    Returns:
    The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        # print('mask calc based on sum')
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
        # print('mask calc based on img')
    return mask_percentage


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    Args:
    np_img: Image as a NumPy array.
    Returns:
    The percentage of the NumPy array that is tissue.
    """
    return 100 - mask_percent(np_img)
def get_annotation_path(org_path):
    #matching based on num
    slide_num=get_num(org_path)
    match = [s for s in original_file_names if slide_num in s][0]
    
    
#     print("found match:\n %s \n %s"%(match,org_path))
    return match
def get_annotation_image(org_path):
    path = get_annotation_path(org_path)
    return open_image_np(path)
class TissueQuantity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).
    Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.
    Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
    """
    indices = list()
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    for r in range(0, num_row_tiles):
        start_r = r * row_tile_size
        end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
        for c in range(0, num_col_tiles):
            start_c = c * col_tile_size
            end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
            indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
    return indices
def tile_to_pil_tile(tile):
    """
    Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.
    Args:
    tile: Tile object.
    Return:
    Tile as a PIL image.
    """
    t = tile
    slide_filepath = get_slide_path_based_on_PNG_path(t.slide_num,t.test)
    if globals.VERBOSE > 0:
      MessageTools.show_blue("Retrived this slide : %s"%(slide_filepath))
    s = open_slide(slide_filepath)

    x, y = t.o_c_s, t.o_r_s
    #maybe make it just the w h 
    w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
    tile_region = s.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img

def np_tile_to_pil_tile(tile):
    t = tile
# this should be retriving the mask tif slide and getting the required tile out of it, it should 
# match with the tile from the tissue slide 
    slide_filepath =  get_test_annotation_slide_path(t.slide_num)
    MessageTools.show_yellow("Retrived this slide : %s"%(slide_filepath))
    s = open_slide(slide_filepath)
    x, y = t.o_c_s, t.o_r_s
    #maybe make it just the w h 
    w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
    tile_region = s.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img
#TODO  : merge the next two functions into one and make sure uasage is consistent 
def get_slide_path_based_on_PNG_path(path,test=False):
  
  num = get_num(path)
  if globals.VERBOSE > 0:
    MessageTools.show_blue(f" Looking for slide .. {num} in {globals.SLIDES_PATH}")
  if test: 
    fn = NORM_TIFF_DIR
  else : 
    fn = original_file_names
  match = [s for s in fn if (str(num)) in s][0]
  return match
def get_test_annotation_slide_path(path):
  num = get_num(path)
  fn =annotated_tif_test_dir
  match = [s for s in fn if (str(num)) in s][0]
  return match

# def tile_ann_to_pil_tile_(tile):
#     """
#     Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.
#     Args:
#     tile: Tile object.
#     Return:
#     Tile as a PIL image.
#     """
#     t = tile
#     slide_filepath = get_annotation_path(t.slide_num)
#     s = open_slide(slide_filepath)

#     x, y = t.o_c_s, t.o_r_s
#     #maybe make it just the w h 
#     w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
#     tile_region = s.read_region((x, y), 0, (w, h))
#     # RGBA to RGB
#     pil_img = tile_region.convert("RGB")
#     return pil_img
def tile_to_np_tile(tile):
    """
    Convert tile information into the corresponding tile as a NumPy image read from the whole-slide image file.
    Args:
    tile: Tile object.
    Return:
    Tile as a NumPy image.
  """
    pil_img = tile_to_pil_tile(tile)
    np_img = pil_to_np_rgb(pil_img)
    return np_img


def save_display_tile(tile, save=True, display=False,isThereTest=False):
    """
    Save and/or display a tile image.
    Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
    """
    tile_pil_img = tile_to_pil_tile(tile)
    
    _,np_ann = tile.get_np_ann_tile()
    if np_ann is not None:
      np_ann = pil_to_np_rgb(np_ann)
      tile_train_mask = get_tile_train_tile(np_ann)
      tile_train_mask = np_to_pil(tile_train_mask)
    if isThereTest: 
      tile_train_mask = np_tile_to_pil_tile(tile)
    
    if save:
        base = os.path.basename(tile.slide_num)
        root = os.path.dirname(tile.slide_num)
        slide_n = get_num_norm(tile.slide_num)
        
        img_path = os.path.join(root,"train\\"+slide_n+"_"+str(tile.tile_num)+".png")
        mask_path = os.path.join(root,"mask\\"+slide_n+"_"+str(tile.tile_num)+".png")
    dir = os.path.dirname(img_path)
    mask_dir = os.path.dirname(mask_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    tile_pil_img.save(img_path)
    if np_ann is not None:
        tile_train_mask.save(mask_path)
    
    if display:
        tile_pil_img.show()
        tile_train_mask.show()
def get_mask_for_test(path):
  num = get_num(path)
  MessageTools.show_yellow("Getting mask for slide num: %s"%(num))
  match = [s for s in annotated_test_dir if num in s][0]
  np = open_image_np(match)
  return np 


class Tile:
    """
    Class for information about a tile.
    """

    def __init__(self, tile_summary, slide_num, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score,np_ann,test):
        self.tile_summary = tile_summary
        self.slide_num = slide_num
        self.np_scaled_tile = np_scaled_tile
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score
        self.np_ann = np_ann
        self.test = test
    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tile_to_pil_tile(self)

    def get_np_tile(self):
        return tile_to_np_tile(self)

    def save_tile(self):
        save_display_tile(self, save=True, display=False, isThereTest=self.test)

    def display_tile(self):
        save_display_tile(self, save=False, display=True)
    def get_np_ann_tile(self):
        # print (self.np_ann)
        if self.np_ann is not None : 
          return self.np_ann,np_tile_to_pil_tile(self)
        else : 
          return None, None 

#     def display_with_histograms(self):
#         display_tile(self, rgb_histograms=True, hsv_histograms=True)

    def get_np_scaled_tile(self):
        return self.np_scaled_tile

    def get_pil_scaled_tile(self):
        return np_to_pil(self.np_scaled_tile)
def tissue_quantity(tissue_percentage):
    """
    Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.
    Args:
    tissue_percentage: The tile tissue percentage.
    Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
    """
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        return TissueQuantity.HIGH
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        return TissueQuantity.MEDIUM
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        return TissueQuantity.LOW
    else:
        return TissueQuantity.NONE
def score_tile(np_tile, tissue_percent, row, col):
    """
    Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.
    Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.
    Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
    """
    color_factor = hsv_purple_pink_factor(np_tile)
    s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
    amount = tissue_quantity(tissue_percent)
    quantity_factor = tissue_quantity_factor(amount)
    combined_factor = color_factor * s_and_v_factor * quantity_factor
    score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
    return score, color_factor, s_and_v_factor, quantity_factor
def score_tiles(path, np_img=None, dimensions=None, small_tile_in_tile=False,SCALE_FACTOR=SCALE_FACTOR,test=False):
    """
    Score all tiles for a slide and return the results in a TileSummary object.
    Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.
    Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
    In the test condition, take the np of the png of the mask . Then save the tile in the Tile object , when converting to pil for conversion the 
    TIF file fo the mask should be accessed for the full resoultion of the tile.  
    """
    img_path = path
    np_ann = None
    if not test:    
      np_ann = get_mask_focused(path) 
    else: 
 # this gets the mask of the HE tifs from their counter-parts PNGS (ie this the np of a png iamge) not tif 
      # Looks in annotated__test_dir
      np_ann = get_mask_for_test(path)
      MessageTools.show_blue("Using the mask for tissue percentage . . . ")

    if dimensions is None:
        o_h, o_w, h, w = parse_dimensions_from_image_filename(img_path)
    else:
        o_w, o_h, w, h = dimensions
    if np_img is None:
        np_img = open_image_np(img_path)
     
    row_tile_size = round(ROW_TILE_SIZE / SCALE_FACTOR)  # use round?
    col_tile_size = round(COL_TILE_SIZE / SCALE_FACTOR)  # use round?
    MessageTools.show_yellow("size: %d - %d"%(row_tile_size,col_tile_size))
    MessageTools.show_blue(f"Image width: {w} - Hight: {h}")
    num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)
    # if  test:
      # tissue_percent_ = tissue_percent(np_img)
    # else: 
    tissue_percent_ = tissue_percent(np_ann)

    tile_sum = TileSummary(slide_num=path,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=tissue_percent_,
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
    for t in tile_indices:
        count += 1  # tile_num
        r_s, r_e, c_s, c_e, r, c = t
        # print(count,r_s,r_e,c_s,c_e,r,c)
        np_tile = np_img[r_s:r_e, c_s:c_e]
        np_tile_ann = np_ann[r_s:r_e, c_s:c_e]
        t_p = tissue_percent(np_tile_ann)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
            high += 1
        elif amount == TissueQuantity.MEDIUM:
            medium += 1
        elif amount == TissueQuantity.LOW:
            low += 1
        elif amount == TissueQuantity.NONE:
            none += 1
        o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h))
        o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h))

        # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
        if (o_c_e - o_c_s) > COL_TILE_SIZE:
            o_c_e -= 1
        if (o_r_e - o_r_s) > ROW_TILE_SIZE:
            o_r_e -= 1

        score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, r, c)

        np_scaled_tile = np_tile if small_tile_in_tile else None
        tile = Tile(tile_sum, path, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                    o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score,np_tile_ann,test)
        tile_sum.tiles.append(tile)

    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none

    tiles_by_score = tile_sum.tiles_by_score()
    rank = 0
    for t in tiles_by_score:
        rank += 1
        t.rank = rank

    return tile_sum

HSV_PURPLE = 270
HSV_PINK = 330
def hsv_purple_pink_factor(rgb):
    """
    Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
    average is purple versus pink.
    Args:
    rgb: Image an NumPy array.
    Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
    """
    hues = rgb_to_hues(rgb)
    hues = hues[hues >= 260]  # exclude hues under 260
    hues = hues[hues <= 340]  # exclude hues over 340
    if len(hues) == 0:
        return 0  # if no hues between 260 and 340, then not purple or pink
    pu_dev = hsv_purple_deviation(hues)
    pi_dev = hsv_pink_deviation(hues)
    avg_factor = (340 - np.average(hues)) ** 2

    if pu_dev == 0:  # avoid divide by zero if tile has no tissue
        return 0

    factor = pi_dev / pu_dev * avg_factor
    return factor
def rgb_to_hues(rgb):
    """
    Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).
    Args:
    rgb: RGB image as a NumPy array
    Returns:
    1-dimensional array of hue values in degrees
    """
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    h = filter_hsv_to_h(hsv, display_np_info=False)
    return h
def filter_rgb_to_hsv(np_img, display_np_info=True):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).
    Args:
    np_img: RGB image as a NumPy array.
    display_np_info: If True, display NumPy array info and filter time.
    Returns:
    Image as NumPy array in HSV representation.
    """
    hsv = sk_color.rgb2hsv(np_img)
  
    return hsv


def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
    """
    Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
    values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
    https://en.wikipedia.org/wiki/HSL_and_HSV
    Args:
    hsv: HSV image as a NumPy array.
    output_type: Type of array to return (float or int).
    display_np_info: If True, display NumPy array info and filter time.
    Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
    """

    h = hsv[:, :, 0]
    h = h.flatten()
    if output_type == "int":
        h *= 360
        h = h.astype("int")

    return h

def hsv_purple_deviation(hsv_hues):
    """
    Obtain the deviation from the HSV hue for purple.
    Args:
    hsv_hues: NumPy array of HSV hue values.
    Returns:
    The HSV purple deviation.
    """
    purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
    return purple_deviation


def hsv_pink_deviation(hsv_hues):
    """
    Obtain the deviation from the HSV hue for pink.
    Args:
    hsv_hues: NumPy array of HSV hue values.
    Returns:
    The HSV pink deviation.
    """
    pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
    return pink_deviation
def filter_hsv_to_s(hsv):
    """
    Experimental HSV to S (saturation).
    Args:
    hsv:  HSV image as a NumPy array.
    Returns:
    Saturation values as a 1-dimensional NumPy array.
    """
    s = hsv[:, :, 1]
    s = s.flatten()
    return s


def filter_hsv_to_v(hsv):
    """
    Experimental HSV to V (value).
    Args:
    hsv:  HSV image as a NumPy array.
    Returns:
    Value values as a 1-dimensional NumPy array.
    """
    v = hsv[:, :, 2]
    v = v.flatten()
    return v
def hsv_saturation_and_value_factor(rgb):
    """
    Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
    deviations should be relatively broad if the tile contains significant tissue.
    Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png
    Args:
    rgb: RGB image as a NumPy array
    Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
  """
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    s = filter_hsv_to_s(hsv)
    v = filter_hsv_to_v(hsv)
    s_std = np.std(s)
    v_std = np.std(v)
    if s_std < 0.05 and v_std < 0.05:
        factor = 0.4
    elif s_std < 0.05:
        factor = 0.7
    elif v_std < 0.05:
        factor = 0.7
    else:
        factor = 1

    factor = factor ** 2
    return factor
def tissue_quantity_factor(amount):
    """
    Obtain a scoring factor based on the quantity of tissue in a tile.
    Args:
    amount: Tissue amount as a TissueQuantity enum value.
    Returns:
    Scoring factor based on the tile tissue quantity.
    """
    if amount == TissueQuantity.HIGH:
        quantity_factor = 1.0
    elif amount == TissueQuantity.MEDIUM:
        quantity_factor = 0.2
    elif amount == TissueQuantity.LOW:
        quantity_factor = 0.1
    else:
        quantity_factor = 0.0
    return quantity_factor

class TileSummary:
    """
    Class for tile summary information.
    """

    slide_num = None
    orig_w = None
    orig_h = None
    orig_tile_w = None
    orig_tile_h = None
    scale_factor = SCALE_FACTOR
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
        self.slide_num = slide_num
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tiles = []

    def __str__(self):
        return summary_title(self) + "\n" + summary_stats(self)

    def mask_percentage(self):
        """
        Obtain the percentage of the slide that is masked.
        Returns:
           The amount of the slide that is masked as a percentage.
        """
        return 100 - self.tissue_percentage

    def num_tiles(self):
        """
        Retrieve the total number of tiles.
        Returns:
          The total number of tiles (number of rows * number of columns).
        """
        return self.num_row_tiles * self.num_col_tiles

    def tiles_by_tissue_percentage(self):
        """
        Retrieve the tiles ranked by tissue percentage.
        Returns:
           List of the tiles ranked by tissue percentage.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
        return sorted_list

    def tiles_by_score(self):
        """
        Retrieve the tiles ranked by score.
        Returns:
           List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list

    def top_tiles(self):
        """
        Retrieve the top-scoring tiles.
        Returns:
           List of the top-scoring tiles.
        """
        sorted_tiles = self.tiles_by_score()
        top_tiles = sorted_tiles[:NUM_TOP_TILES]
        return top_tiles

    def get_tile(self, row, col):
        """
        Retrieve tile by row and column.
        Args:
          row: The row
          col: The column
        Returns:
          Corresponding Tile object.
        """
        tile_index = (row - 1) * self.num_col_tiles + (col - 1)
        tile = self.tiles[tile_index]
        return tile

    def display_summaries(self):
        """
        Display summary images.
        """
        summary_and_tiles(self.slide_num, display=True, save_summary=False, save_data=False, save_top_tiles=False)
        
def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
    """
    Create a PIL summary image including top title area and right side and bottom padding.
    Args:
    np_img: Image as a NumPy array.
    title_area_height: Height of the title area at the top of the summary image.
    row_tile_size: The tile size in rows.
    col_tile_size: The tile size in columns.
    num_row_tiles: The number of row tiles.
    num_col_tiles: The number of column tiles.
    Returns:
    Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
    potentially a top title area and right side and bottom padding.
    """
    r = row_tile_size * num_row_tiles + title_area_height
    c = col_tile_size * num_col_tiles
    print(f"number of rows,cols for summary : {r,c}")
    summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
    # add gray edges so that tile text does not get cut off
    summary_img.fill(120)
    # color title area white
    summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
    summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
    summary = np_to_pil(summary_img)
    return summary
def faded_tile_border_color(tissue_percentage):
    """
    Obtain the corresponding faded tile border color for a particular tile tissue percentage.
    Args:
    tissue_percentage: The tile tissue percentage
    Returns:
    The faded tile border color corresponding to the tile tissue percentage.
    """
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        border_color = FADED_THRESH_COLOR
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        border_color = FADED_MEDIUM_COLOR
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        border_color = FADED_LOW_COLOR
    else:
        border_color = FADED_NONE_COLOR
    return border_color
def tile_border_color(tissue_percentage):
    """
    Obtain the corresponding tile border color for a particular tile tissue percentage.
    Args:
    tissue_percentage: The tile tissue percentage
    Returns:
    The tile border color corresponding to the tile tissue percentage.
    """
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        border_color = HIGH_COLOR
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        border_color = MEDIUM_COLOR
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        border_color = LOW_COLOR
    else:
        border_color = NONE_COLOR
    return border_color
def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
    """
    Draw a border around a tile with width TILE_BORDER_SIZE.
    Args:
    draw: Draw object for drawing on PIL image.
    r_s: Row starting pixel.
    r_e: Row ending pixel.
    c_s: Column starting pixel.
    c_e: Column ending pixel.
    color: Color of the border.
    border_size: Width of tile border in pixels.
  """
    for x in range(0, border_size):
        draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

def add_tile_stats_to_top_tile_summary(pil_img, tiles, z):
    np_sum = pil_to_np_rgb(pil_img)
    sum_r, sum_c, sum_ch = np_sum.shape
    np_stats = np_tile_stat_img(tiles)
    st_r, st_c, _ = np_stats.shape
    combo_c = sum_c + st_c
    combo_r = max(sum_r, st_r + z)
    combo = np.zeros([combo_r, combo_c, sum_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:sum_r, 0:sum_c] = np_sum
    combo[z:st_r + z, sum_c:sum_c + st_c] = np_stats
    result = np_to_pil(combo)
    return result

def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
    """
    Obtain a PIL image representation of text.
    Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.
    Returns:
    PIL image representing the text.
    """

    font = ImageFont.truetype(font_path, font_size)
    x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
    image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
    draw = ImageDraw.Draw(image)
    draw.text((w_border, h_border), text, text_color, font=font)
    return image
def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
    """
    Obtain a NumPy array image representation of text.
    Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.
    Returns:
    NumPy array representing the text.
    """
    pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                     text_color, background)
    np_img = pil_to_np_rgb(pil_img)
    return np_img
def summary_stats(tile_summary):
    """
    Obtain various stats about the slide tiles.
    Args:
    tile_summary: TileSummary object.
    Returns:
     Various stats about the slide tiles as a string.
    """
    return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)
def np_tile_stat_img(tiles):
    """
    Generate tile scoring statistics for a list of tiles and return the result as a NumPy array image.
    Args:
    tiles: List of tiles (such as top tiles)
    Returns:
    Tile scoring statistics converted into an NumPy array image.
    """
    tt = sorted(tiles, key=lambda t: (t.r, t.c), reverse=False)
    tile_stats = "Tile Score Statistics:\n"
    count = 0
    for t in tt:
        if count > 0:
            tile_stats += "\n"
    count += 1
    tup = (t.r, t.c, t.rank, t.tissue_percentage, t.color_factor, t.s_and_v_factor, t.quantity_factor, t.score)
    tile_stats += "R%03d C%03d #%003d TP:%6.2f%% CF:%4.0f SVF:%4.2f QF:%4.2f S:%0.4f" % tup
    np_stats = np_text(tile_stats, font_path=SUMMARY_TITLE_FONT_PATH, font_size=14)
    return np_stats
def generate_top_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=True,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):
    """
    Generate summary images/thumbnails showing the top tiles ranked by score.
    Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display top tiles to screen.
    save_summary: If True, save top tiles images.
    show_top_stats: If True, append top tile score stats to image.
    label_all_tiles: If True, label all tiles. If False, label only top tiles.
    """
    np_img=open_image_np(np_img)
    num_slide = get_num(tile_sum.slide_num)
    z = 300  # height of area at top of summary slide
    slide_num = tile_sum.slide_num
    rows = tile_sum.scaled_h
    cols = tile_sum.scaled_w
    row_tile_size = tile_sum.scaled_tile_h
    col_tile_size = tile_sum.scaled_tile_w
    print(row_tile_size,col_tile_size)
    num_row_tiles, num_col_tiles = tile_sum.num_row_tiles,tile_sum.num_col_tiles   #get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size ,num_col_tiles,num_row_tiles)# changed the order of positional params ,num_row_tiles ,num_col_tiles,--->num_col_tiles,,num_row_tiles 
    draw = ImageDraw.Draw(summary)

    np_orig = open_image_np(slide_num)
    summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size,  num_col_tiles,num_row_tiles)
    draw_orig = ImageDraw.Draw(summary_orig)

    if border_all_tiles:
        for t in tile_sum.tiles:
            border_color = faded_tile_border_color(t.tissue_percentage)
            tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
            tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

    tbs = TILE_BORDER_SIZE
    top_tiles = tile_sum.top_tiles()
    for t in top_tiles:
        border_color = tile_border_color(t.tissue_percentage)
        tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        if border_all_tiles:
            tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
            tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

    summary_title = "Slide %0s Top Tile Summary:" % num_slide
    summary_txt = summary_title + "\n" + summary_stats(tile_sum)

    summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
    draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
    draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

    tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
    h_offset = TILE_BORDER_SIZE + 2
    v_offset = TILE_BORDER_SIZE
    h_ds_offset = TILE_BORDER_SIZE + 3
    v_ds_offset = TILE_BORDER_SIZE + 1
    for t in tiles_to_label:
        label = "R%d\nC%d" % (t.r, t.c)
        font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
        # drop shadow behind text
        draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
        draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

        draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
        draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

    if show_top_stats:
        summary = add_tile_stats_to_top_tile_summary(summary, top_tiles, z)
        summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, top_tiles, z)

    if display:
        summary.show()
        summary_orig.show()
    return summary, summary_orig

def apply_image_filters(np_img,path, slide_num=None, info=None, save=False, display=False):
  """
  Apply filters to image as NumPy array and optionally save and/or display filtered images.
  Args:
    np_img: Image as NumPy array.
    slide_num: The slide number (used for saving/displaying).
    info: Dictionary of slide information (used for HTML display).
    save: If True, save image.
    display: If True, display image.
  Returns:
    Resulting filtered image as a NumPy array.
  """
  rgb = np_img

  mask_not_green = filter_green_channel(rgb)
  rgb_not_green = mask_rgb(rgb, mask_not_green)

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = mask_rgb(rgb, mask_not_gray)

  mask_no_red_pen = filter_red_pen(rgb)
  rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)

  mask_no_green_pen = filter_green_pen(rgb)
  rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)

  mask_no_blue_pen = filter_blue_pen(rgb)
  rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)

  mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
  rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
  rgb_remove_small = mask_rgb(rgb, mask_remove_small)

  img = rgb_remove_small

  print(os.path.exists(FILTER_DIR))
  if save and not os.path.exists(FILTER_DIR):
    os.makedirs(FILTER_DIR)
  if save : 
    save_np(img,path)
  return img

def apply_filters_to_image(slide_num, save=True, display=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.
  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
  Returns:
    Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
    (used for HTML page generation).
  """
  MessageTools.show_yellow("Processing slide #%d" % slide_num)

  info = dict()

  if save and not os.path.exists(FILTER_DIR):
    os.makedirs(FILTER_DIR)
  img_path = (slide_num)
  np_orig =open_image_np(img_path)
  filtered_np_img = apply_image_filters(np_orig, slide_num, info, save=save, display=display)

#   if save:
#     result_path =get_filter_image_result(slide_num)
#     pil_img = np_to_pil(filtered_np_img)
#     pil_img.save(result_path)
#     print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

#     thumbnail_path = get_filter_thumbnail_result(slide_num)
#     save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_path)
#     print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

  print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  return filtered_np_img, info


def filter_green_channel(np_img, green_thresh=228, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.
  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    print(
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.
  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
  """
  Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
  Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
  lower threshold value rather than a blue channel upper threshold value.
  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.
  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.
  Returns:
    NumPy array representing the mask.
  """
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.
  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing the mask.
  """
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  return result


def filter_grays(rgb, tolerance=13, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.
  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    
    result = result.astype("uint8") * 255
  return result


def uint8_to_bool(np_img):
  """
  Convert NumPy array of uint8 (255,0) values to bool (True,False) values
  Args:
    np_img: Binary image as NumPy array of uint8 (255,0) values.
  Returns:
    NumPy array of bool (True,False) values.
  """
  result = (np_img / 255).astype(bool)
  return result
def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.
  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8).
  """

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    MessageTools.show_blue("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  return np_img

def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.
  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  result = rgb * np.dstack([mask, mask, mask])
  return result

def save_np(np_img,path):
  base  = os.path.basename(path)
  
  save_path = os.path.join(FILTER_DIR,base)

  img =  np_to_pil(np_img)
  img.save(save_path)


import spams
import cv2 as cv

class TissueMaskException(Exception):
    pass

######################################################################################################

def is_uint8_image(I):
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True
######################################################################################################

def is_image(I):
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True
######################################################################################################

def get_tissue_mask(I, luminosity_threshold=0.8):
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
    if mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")

    return mask

######################################################################################################

def convert_RGB_to_OD(I):
    mask = (I == 0)
    I[mask] = 1
    

    #return np.maximum(-1 * np.log(I / 255), 1e-6)
    return np.maximum(-1 * np.log(I / 255), np.zeros(I.shape) + 0.1)

######################################################################################################

def convert_OD_to_RGB(OD):
    
    assert OD.min() >= 0, "Negative optical density."
    
    OD = np.maximum(OD, 1e-6)
    
    return (255 * np.exp(-1 * OD)).astype(np.uint8)

######################################################################################################

def normalize_matrix_rows(A):
    return A / np.linalg.norm(A, axis=1)[:, None]

######################################################################################################


def get_concentrations(I, stain_matrix, regularizer=0.01):
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T

######################################################################################################

def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):

    # Convert to OD and ignore background
    tissue_mask = get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    
    OD = OD[tissue_mask]

    # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

    # The two principle eigenvectors
    V = V[:, [2, 1]]

    # Make sure vectors are pointing the right way
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    # Project on this basis.
    That = np.dot(OD, V)

    # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])

    # Min and max angles
    minPhi = np.percentile(phi, 100 - angular_percentile)
    maxPhi = np.percentile(phi, angular_percentile)

    # the two principle colors
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Order of H and E.
    # H first row.
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])

    return normalize_matrix_rows(HE)

######################################################################################################

def mapping(target,source):
    
    stain_matrix_target = get_stain_matrix(target)
    target_concentrations = get_concentrations(target,stain_matrix_target)
    maxC_target = np.percentile(target_concentrations, 99, axis=0).reshape((1, 2))
    stain_matrix_target_RGB = convert_OD_to_RGB(stain_matrix_target) 
    
    stain_matrix_source = get_stain_matrix(source)
    source_concentrations = get_concentrations(source, stain_matrix_source)
    maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    source_concentrations *= (maxC_target / maxC_source)
    tmp = 255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target))
    return tmp.reshape(source.shape).astype(np.uint8)


#### Utils for predictions ####
def get_y(im_f):
  msk_path = os.path.join(os.path.dirname(annotated_file_names[0]),os.path.basename(im_f))
#   msk = np.array(PILMask.create(msk_path))
#   msk[msk==255]=1
  # mask = msk.astype("int8")
  return msk_path 
def get_y_test(fn):
    base=os.path.basename(fn)
    base=os.path.splitext(base)[0]
    match = [s for s in both_normal_tum if base in s]
    return match[0]
def n_codes(fnames, is_partial=True):
  "Gather the codes from a list of `fnames`"
  vals = set()
  if is_partial:
    random.shuffle(fnames)
    fnames = fnames[:10]
  for fname in fnames:
    msk = np.array(PILMask.create(fname))
    for val in np.unique(msk):
      if val not in vals:
        vals.add(val)
  vals = list(vals)
  p2c = dict()
  for i,val in enumerate(vals):
    p2c[i] = vals[i]
  return p2c  
def get_msk(fn, p2c):
  "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
  fn = get_y(fn)
  msk = np.array(PILMask.create(fn))
  mx = np.max(msk)
  
  for i, val in enumerate(p2c):
    msk[msk==p2c[i]] = val
  MessageTools.show_blue(np.unique(msk))
  msk=msk.astype("bool")
  return PILMask.create(msk).convert("1")
def tumour_norm(input, target):
#input is pred (2,256,256)
    target = target.squeeze(1)
  # target is ground truth (256,256)
    if len(torch.unique(target))>5:
        mask = target !=2
    else :
        mask = target!=1
    # print(mask)
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def not_tumour(input, target):
#input is pred (2,256,256)
  target = target.squeeze(1)
  # target is ground truth (256,256)
  mask = target == void_code
  # print(mask)
  return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
def segmentron_splitter(model):
  return [params(model.backbone), params(model.head)]
## Return Jaccard index, or Intersection over Union (IoU) value
def IoU_loss(preds:Tensor, targs:Tensor, eps:float=1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value 
             for multi-class image segmentation
    """
    num_classes = preds.shape[1]
    
    # Single class segmentation?
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(preds)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        
    # Multi-class segmentation
    else:
        # Convert target to one-hot encoding
        # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
        true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]
        
        # Permute [B,H,W,C] to [B,C,H,W]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # Take softmax along class dimension; all class probs add to 1 (per pixel)
        probas = F.softmax(preds, dim=1)
        
    true_1_hot = true_1_hot.type(preds.type())
    
    # Sum probabilities by class and across batch images
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
    cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
    return 1-iou
#Return Jaccard index, or Intersection over Union (IoU) value
def IoU(preds:Tensor, targs:Tensor, eps:float=1e-8):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value 
             for multi-class image segmentation
    """
    num_classes = preds.shape[1]
    
    # Single class segmentation?
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(preds)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        
    # Multi-class segmentation
    else:
        # Convert target to one-hot encoding
        # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
        true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]
        
        # Permute [B,H,W,C] to [B,C,H,W]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # Take softmax along class dimension; all class probs add to 1 (per pixel)
        probas = F.softmax(preds, dim=1)
        
    true_1_hot = true_1_hot.type(preds.type())
    
    # Sum probabilities by class and across batch images
    dims = (0,) + tuple(range(2, targs.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
    cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
    union = cardinality - intersection
    iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
    return iou
def seg_accuracy(input:Tensor, targs:Tensor):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()


# %%
class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0
    def __init__(self, aug): 
        self.aug = aug
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])
class TargetMaskConvertTransform(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        mask = np.array(mask)
        
        li = (np.unique(mask))
        if 29 in li:
          mask[mask==29]=0
        # normal case
        if len(li)>5:
            mask[mask!=255]=0
            mask[mask==255]=2
        #tumour
        else:
            mask[mask!=255]=0
            mask[mask==255]=1
        mask = PILMask.create(mask)
        return img, mask
class TargetMaskConvertTransform_test(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        mask = np.array(mask)

       
    # Change 255 for 1
        mask[mask==1]=0
        mask[mask==2]=1

        # Back to PILMask
        mask = PILMask.create(mask)
        return img, mask
class SN(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        img=np.array(img)
        #Normalise patch according to target
        img = mapping(target,img)

     
        img=PILImage.create(img)
        return img, mask
from dataio.loader.cc_dataset import cc_dataset
from dataio.loader.asdc_dataset import asdc_dataset
from dataio.loader.monuseg_dataset import monuseg_dataset
import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.stain_norm_dataset import stain_norm_dataset
from dataio.loader.slides_dataset import slides_dataset
from dataio.loader.peso_dataset import peso_dataset
from dataio.loader.glas_dataset import glas_dataset
from dataio.loader.siim_acr_dataset import siim_acr_dataset
from dataio.loader.monuseg_dataset import monuseg_dataset
from dataio.loader.isic_dataset import isic_dataset


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'ukbb_sax': CMR3DDataset,
        'acdc_sax': CMR3DDataset,
        'rvsc_sax': CMR3DDataset,
        'hms_sax':  HMSDataset,
        'test_sax': TestDataset,
        'us': UltraSoundDataset,
        'epi':stain_norm_dataset,
        'epi_slides':slides_dataset,
        'peso':peso_dataset,
        'pesoL':peso_dataset,
        'glas':glas_dataset,
        'siim':siim_acr_dataset,
        'monuseg':monuseg_dataset,
        'isic':isic_dataset,
        'asdc':asdc_dataset,
        'cc':cc_dataset,

    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """
    return {
     "epi": "kaggle/input/stain-normalisation/sn",
      "epi_slides":"c:\\users\\usama\\codeserverepi\\codeserverepi-colab\\",
      "peso":"F:\\data\\peso_dataset",
      "pesol":"F:\\data\\peso_dataset\\scaled_slides_tif",
      "glas":"F:\\data\\warwick qu dataset (released 2016_07_08)",
      "siim":"F:\\data\\siim acr",
      "monuseg":"c:\\data\\monuseg",
      "isic":"F:\\data\\isic"
    }[dataset_name]
    # return getattr(opts, dataset_name)

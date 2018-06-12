#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:22:34 2018

@author: hyang
"""
from argparse import ArgumentParser
import segmentDeepLab as seg

# !!! Notes from Matt RE: licenses !!!
# 3 main licenses: apache, mit, gnu (!!! need to ask for express permissions), add docstring in each function citing repo

# Main script to run fast deep photo style transfer

#%% Define defaults
main_dir = '/Users/hyang/Work/Insight/fast-deep-photo-style-transfer-tf/'
model_dir = main_dir + 'deeplab/models/'
# Default folders for DeepLab
input_dir = main_dir + 'inputPhotos/'
resized_dir = main_dir + 'resizedPhotos/'
style_dir = main_dir + 'stylePhotos/'
seg_dir = main_dir + 'segMaps/'
output_dir = main_dir + 'outputPhotos/'
# Default folders for fast style transfer
VGG_PATH = 'fast-style-transfer-tf/data/imagenet-vgg-verydeep-19.mat' # point to deep photo weights
CHECKPOINT_DIR = 'fast-style-transfer-tf/checkpoints/'

## Parser function
#def build_parser():
#    """Parser function"""
#    parser = ArgumentParser()
#    parser.add_argument('--checkpoint-dir', type=str,
#                        dest='checkpoint_dir', help='dir to save checkpoint in',
#                        metavar='CHECKPOINT_DIR', required=True)
#
#    parser.add_argument('--style', type=str,
#                        dest='style', help='style image path',
#                        metavar='STYLE', required=True)
#
#    parser.add_argument('--train-path', type=str,
#                        dest='train_path', help='path to training images folder',
#                        metavar='TRAIN_PATH', default=TRAIN_PATH)
#
#    parser.add_argument('--indir', type=str,
#                        dest='in_dir', help='Input image path',
#                        metavar='TEST', default=False)
#
#    parser.add_argument('--outdir', type=str,
#                        dest='out_dir', help='Output styled image save dir',
#                        metavar='TEST_DIR', default=False)
#
#    parser.add_argument('--vgg-path', type=str,
#                        dest='vgg_path',
#                        help='path to VGG19 network (default %(default)s)',
#                        metavar='VGG_PATH', default=VGG_PATH)
#
#    return parser

# %% Parser function
def build_parser():
    """Parser function"""
    parser = ArgumentParser()

    parser.add_argument('--in_dir', type=str,
                        dest='input_dir', help='Input image directory',
                        metavar='INPUT_DIR', default=input_dir)

    parser.add_argument('--resized_dir', type=str,
                        dest='resized_dir', help='Resized image directory',
                        metavar='RESIZED_DIR', default=input_dir)

    parser.add_argument('--style_dir', type=str,
                        dest='style_dir', help='Style image directory',
                        metavar='STYLE_DIR', default=style_dir)

    parser.add_argument('--seg_dir', type=str,
                        dest='style_dir', help='Style image directory',
                        metavar='SEG_DIR', default=seg_dir)

    parser.add_argument('--out_dir', type=str,
                        dest='out_dir', help='Output styled image save directory',
                        metavar='TEST_DIR', default=output_dir)

    parser.add_argument('--input', type=str,
                        dest='inputName', help='Input image file name, including extension',
                        metavar='INPUTNAME', required=True)

    parser.add_argument('--style', type=str,
                        dest='styleName', help='Style image file name, including extension',
                        metavar='STYLENAME', required=True)
    return parser
# %% Helper methods

def ensure_folders(directory):
    """ If directory doesn't exist, make directory """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def check_options(options):
    options.inputFile = options.input_dir + options.inputName
    options.styleFile = options.style_dir + options.styleName
    
    ensure_folders(input_dir)
    ensure_folders(resized_dir)
    ensure_folders(style_dir)
    ensure_folders(seg_dir)
    ensure_folders(output_dir)
    #
    # !!! IF NAMES MATCH, THROW EXCEPTION !!!
    if options.inputName == options.styleName:
        raise ValueError('Input and style file names cannot be the same')
    return options

# Function to retrieve files from directory
def _get_files(img_dir):
    """List all files in directory"""
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]


# %% Pipeline for FST ONLY !!!!!!!!!!!!!!

#    parser = build_parser()
#    options = parser.parse_args()
#    options = check_options(options)

# Temporary option testing
parser = ArgumentParser()
options = parser.parse_args()

# Input directories
options.inputName = 'image2.jpg'
options.styleName = 'leopard.jpg'
options.input_dir = main_dir + 'inputPhotos/'
options.resized_dir = main_dir + 'resizedPhotos/'
options.style_dir = main_dir + 'stylePhotos/'
options.seg_dir = main_dir + 'segMaps/'
options.output_dir = main_dir + 'outputPhotos/'
options.inputFile = options.input_dir + options.inputName
options.styleFile = options.style_dir + options.styleName

seg.main(options)

#python evaluate.py --checkpoint path/to/style/model.ckpt \
#  --in-path dir/of/test/imgs/ \
#  --out-path dir/for/results/

# !python segmentDeepLab.py --input image2.jpg --style leopard.jpg
# Segment input photo and store
    
# Style photo
    
# Segment style photo and store
    
# Feed input, segmap, style photo, segmap into FPST
    

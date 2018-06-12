#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:22:34 2018

@author: hyang
"""
import os, errno
from argparse import ArgumentParser
import segmentDeepLab as seg
import fst
from fast_style_transfer.src.utils import exists
from subprocess import call

# !!! Notes from Matt RE: licenses !!!
# 3 main licenses: apache, mit, gnu (!!! need to ask for express permissions), add docstring in each function citing repo

# Main script to run fast deep photo style transfer

#%% Define defaults
main_dir = '/Users/hyang/Work/Insight/fast-deep-photo-style-transfer-tf/'
model_path = main_dir + 'deeplab/models/deeplab_model.tar.gz'
# Default folders for DeepLab
input_dir = main_dir + 'inputPhotos/'
resized_dir = main_dir + 'resizedPhotos/'
style_dir = main_dir + 'stylePhotos/'
seg_dir = main_dir + 'segMaps/'
output_dir = main_dir + 'outputPhotos/'
# Default folders for fast style transfer
VGG_PATH = 'fast-style-transfer-tf/data/imagenet-vgg-verydeep-19.mat' # point to deep photo weights
# FST options
BATCH_SIZE = 4
DEVICE = '/gpu:0'

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

#python run_fpst.py --in-path inputPhotos/insightCorner.jpg --style-path stylePhotos/leopard.jpg --checkpoint-path checkpoints/udnie.ckpt

# %% Parser function
def build_parser():
    """Parser function"""
    parser = ArgumentParser()

# Input and output
    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='Input image path',
                        metavar='IN_PATH', required=True)

    parser.add_argument('--style-path', type=str,
                        dest='style_path', help='Style image path',
                        metavar='STYLE_PATH', required=True)

    # Default output path to same name as input in parent directory
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help='Output styled image path',
                        metavar='OUT_PATH', required=True)
    
# Intermediate file save directories
    parser.add_argument('--resized_dir', type=str,
                        dest='resized_dir', help='Resized image directory',
                        metavar='RESIZED_DIR', default=resized_dir)

    parser.add_argument('--seg_dir', type=str,
                        dest='seg_dir', help='Segmented image directory',
                        metavar='SEG_DIR', default=seg_dir)

# Deep Lab
    parser.add_argument('--model-path', type=str,
                        dest='model_path', help='Path to DeepLab model',
                        metavar='MODEL_path', default=model_path)

# Fast style transfer
    parser.add_argument('--checkpoint-path', type=str,
                        dest='checkpoint_dir', # TEMPORARILY MAINTAIN FST NAMING CONVENTION
                        help='Directory containing checkpoint files',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser
# %% Helper methods

def ensure_folders(directory):
    """ If directory doesn't exist, make directory """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def check_opts(opts):
    opts.inputFileName = opts.in_path.split('/')[-1]#.split('.')[0]
    opts.styleFileName = opts.style_path.split('/')[-1]#.split('.')[0]
    opts.checkpointName = opts.checkpoint_dir.split('/')[-1].split('.')[0]
    opts.resized_path = opts.resized_dir + opts.inputFileName
    
    ensure_folders(input_dir)
    ensure_folders(resized_dir)
    ensure_folders(style_dir)
    ensure_folders(seg_dir)
    ensure_folders(output_dir)
    #
    # !!! IF NAMES MATCH, THROW EXCEPTION !!!
    if opts.inputFileName == opts.styleFileName:
        raise ValueError('Input and style file names cannot be the same')
        
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    
    return opts

# Function to retrieve files from directory
def _get_files(img_dir):
    """List all files in directory"""
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

def visualize_result(opts)

# %% Pipeline for FST ONLY !!!!!!!!!!!!!!
def main():
    parser = build_parser()
    opts = parser.parse_args()
    opts = check_opts(opts)

    # Temporary option testing
    #parser = ArgumentParser()
    #opts = parser.parse_args()
    #
    ## Input directories
    #opts.model_path = model_path
    #opts.inputName = 'insightCorner.jpg'
    #opts.styleName = 'leopard.jpg'
    #opts.checkpointName = 'udnie'
    #opts.checkpoint_dir = 'checkpoints/udnie.ckpt'
    #opts.input_dir = main_dir + 'inputPhotos/'
    #opts.resized_dir = main_dir + 'resizedPhotos/'
    #opts.style_dir = main_dir + 'stylePhotos/'
    #opts.seg_dir = main_dir + 'segMaps/'
    #opts.output_dir = main_dir + 'outputPhotos/'
    #opts.in_path = opts.input_dir + opts.inputName
    #opts.resized_path = opts.resized_dir + opts.inputName
    #opts.style_path = opts.style_dir + opts.styleName
    #opts.out_path = opts.output_dir + opts.checkpointName + '_' + opts.inputName
    #opts.device = '/gpu:0'
    
    # Call DeepLab auto-segmentation
    seg.main(opts)
    # Call Logan Engstrom's fast style transfer
    fst.main(opts)
    
    call(['open' , opts.out_path])
    
    #python evaluate.py --checkpoint path/to/style/model.ckpt \
    #  --in-path dir/of/test/imgs/ \
    #  --out-path dir/for/results/
    
    # !python segmentDeepLab.py --input image2.jpg --style leopard.jpg
    # Segment input photo and store
        
    # Style photo
        
    # Segment style photo and store
        
    # Feed input, segmap, style photo, segmap into FPST
    
if __name__ == '__main__':
    main()
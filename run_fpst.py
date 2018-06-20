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

# Profile slow deep photo style transfer
import cProfile

# !!! Notes from Matt RE: licenses !!!
# 3 main licenses: apache, mit, gnu (!!! need to ask for express permissions), add docstring in each function citing repo

# Main script to run fast deep photo style transfer

#%% Define defaults
main_dir = '/Users/hyang/Work/Insight/fast-deep-photo-style-transfer-tf/'
deeplab_path = os.path.join(main_dir, 'deeplab/models/deeplab_model.tar.gz')
# Default folders for DeepLab
input_dir = os.path.join(main_dir, 'inputPhotos/')
resized_dir = os.path.join(main_dir, 'resizedPhotos/')
style_dir = os.path.join(main_dir + 'stylePhotos/')
seg_dir = os.path.join(main_dir + 'segMaps/')
output_dir = os.path.join(main_dir + 'outputPhotos/')
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
    parser.add_argument('--resized-dir', type=str,
                        dest='resized_dir', help='Resized image directory',
                        metavar='RESIZED_DIR', default=resized_dir)

    parser.add_argument('--seg-dir', type=str,
                        dest='seg_dir', help='Segmented image directory',
                        metavar='SEG_DIR', default=seg_dir)

# Deep Lab
    parser.add_argument('--deeplab-path', type=str,
                        dest='deeplab_path', help='Path to DeepLab model',
                        metavar='DEEPLAB_path', default=deeplab_path)

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
    
    # Deep photo style transfer (slow)
    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='Original Luan approach (very slow)',
                        default=False)

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
    opts.resized_dir = os.path.abspath(opts.resized_dir)
    opts.seg_dir = os.path.abspath(opts.seg_dir)
    opts.deeplab_path = os.path.abspath(opts.deeplab_path)
    
    opts.inputFileName = opts.in_path.split('/')[-1]#.split('.')[0]
    opts.styleFileName = opts.style_path.split('/')[-1]#.split('.')[0]
    opts.checkpointName = opts.checkpoint_dir.split('/')[-1].split('.')[0]
    opts.resized_path = os.path.join(opts.resized_dir, opts.inputFileName)
    opts.resized_style_path = os.path.join(opts.resized_dir, opts.styleFileName)
    opts.seg_path = os.path.join(opts.seg_dir + opts.inputFileName)
    opts.seg_style_path = os.path.join(opts.seg_dir, opts.styleFileName)
    
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

# %% Pipeline for FST ONLY !!!!!!!!!!!!!!
def main():
    parser = build_parser()
    opts = parser.parse_args()
    opts = check_opts(opts)
    
    # Call DeepLab auto-segmentation
    #if segment
    
###################################    seg.main(opts)

    seg.main(opts.deeplab_path, opts.in_path, opts.inputFileName, opts.resized_dir, opts.seg_dir)
    seg.main(opts.deeplab_path, opts.style_path, opts.styleFileName, opts.resized_dir, opts.seg_dir)
    # Call Logan Engstrom's fast style transfer
    #if train:
    
    ### !!!call training function here
    
    #else:
    
    if opts.slow:
        print("CALLING SLOW DEEP PHOTO STYLE")
        print("Slow: %s" % opts.slow)
        cmd = ['python', '-m', 'cProfile', '-o', 'deepPhotoProfile_Adams' \
        , 'deep-photo-styletransfer-tf/deep_photostyle.py', '--content_image_path' \
        , opts.resized_path, '--style_image_path', opts.resized_style_path \
        , '--content_seg_path', opts.seg_path, '--style_seg_path', opts.seg_style_path \
        , '--style_option', '2', '--output_image', opts.out_path \
        , '--max_iter', '100', '--save_iter', '5', '--lbfgs']
        call(cmd)
    else:
        print("CALLING FAST STYLE TRANSFER")
        fst.main(opts)
#        python deep_photostyle.py --content_image_path ./test/input/resized_im.jpg
#        --style_image_path ./test/style/resized_leopard.jpg --content_seg_path
#        ./test/seg/seg_map.jpg --style_seg_path ./test/seg/seg_leopard.jpg --style_option 2

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

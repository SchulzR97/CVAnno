import argparse

def get_args():
    parser = argparse.ArgumentParser('CVAnno')
    parser.add_argument('--annotation_dir', type=str)
    parser.add_argument('--type', type=str, default='segmentation')
    
    args = parser.parse_args()
    return args
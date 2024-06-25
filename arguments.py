import argparse

def get_args():
    parser = argparse.ArgumentParser('CVAnno')
    parser.add_argument('--annotation_dir', type=str)
    
    args = parser.parse_args()
    return args
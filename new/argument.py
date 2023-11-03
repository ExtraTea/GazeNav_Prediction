import argparse

def get_args():
    parser = argparse.ArgumentParser(description='rl')

    parser.add_argument('--output_dir', type = str, default='trainedmodels/my_model')
    parser.add_argument('--seed', type = int, default=0)
    parser
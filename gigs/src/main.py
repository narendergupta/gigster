from datamodel import DataModel
from experimenter import Experimenter
from gen_utils import *

import argparse


def main(args):
    dm = DataModel(args.gig_file, args.chat_file)
    dm.read_data()
    exp = Experimenter(dm)
    if args.classify is True:
        scores = exp.classify_gigs()
    if args.feature_values is True:
        scores = exp.evaluate_feature_values()
    return dm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gig_file", required=True)
    parser.add_argument("--chat_file", required=True)
    parser.add_argument("--classify", action='store_true', default=False)
    parser.add_argument("--feature_values", action='store_true', default=False)
    args = parser.parse_args()
    dm = main(args)


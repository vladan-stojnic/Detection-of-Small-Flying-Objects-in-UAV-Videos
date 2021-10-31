# Copyright (c) 2021 Project Bee4Exp.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Defines utility functions.
"""


def get_parser():
    """Get parser object."""

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",
                        dest="input",
                        help="Path to input",
                        required=False)
    parser.add_argument("-o", "--output",
                        dest="output",
                        help="Path to output",
                        required=False)
    parser.add_argument("--mask",
                        dest="mask",
                        help="Path to directory with mask files",
                        required=False)
    parser.add_argument("--bee_mean",
                        dest="bee_mean",
                        help="""Value of mean for synthetic honeybees. For choice enter as 0.25,0.5 and
                                for uniform range enter [0.25,0.5].""",
                        default="0.25",
                        type=str,
                        required=False)
    parser.add_argument("--num_synthetic_videos",
                        dest="num_synth_videos",
                        help="Number of synthetic videos to generate.",
                        default=1000,
                        type=int,
                        required=False)
    parser.add_argument("--num_avg",
                        dest="num_avg",
                        help="Number of frames used for averaging in background subtraction.",
                        default=50,
                        type=int,
                        required=False)
    parser.add_argument("--type",
                        dest="type",
                        help="Type of HDF5 dataset.",
                        default="train",
                        choices=["train", "val", "test"],
                        required=False)
    parser.add_argument("-a", "--annot",
                        dest="annot",
                        help="Npy file with human annotations.")
    parser.add_argument("--train_data",
                        dest="train_data",
                        help="Train data",
                        required=False)
    parser.add_argument("--val_data",
                        dest="val_data",
                        help="Validation data",
                        required=False)
    parser.add_argument("--test_data",
                        dest="test_data",
                        help="Test data",
                        required=False)
    parser.add_argument("--model",
                        dest="model",
                        help="Model path",
                        required=False)
    parser.add_argument("--thr",
                        dest="thr",
                        help="Threshold",
                        type=float,
                        default=0.1,
                        required=False)
    parser.add_argument("--heat_map",
                        dest="heat_map",
                        help="Path to heatmap",
                        required=False)
    return parser

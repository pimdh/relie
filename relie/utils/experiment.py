from PIL import Image
import numpy as np
import torch
import matplotlib
import os
import sys
from os import environ
import logging
from pprint import pformat
import json
import secrets
from datetime import datetime
from tensorboardX import SummaryWriter
from subprocess import check_output

if not environ.get('DISPLAY', ''):
    matplotlib.use("Agg")


def setup_logging(path):
    formatter = logging.Formatter("%(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers = []

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)


def setup_experiment(experiment_name, run_name, args):
    if 'RESULTS_PATH' in environ:
        out_dir = environ['RESULTS_PATH']
    else:
        out_dir = './outputs'

    t = str(int(datetime.utcnow().timestamp())) + secrets.token_hex(2)
    identifier = f"{run_name}-{t}" if run_name else t

    def out_path(category=None, filename=None):
        path = os.path.join(out_dir, experiment_name, identifier)
        if category is not None:
            path = os.path.join(path, category)
        os.makedirs(path, exist_ok=True)
        if filename is not None:
            return os.path.join(path, filename)
        return path

    setup_logging(out_path(filename='out.log'))

    commit = check_output(['git', 'describe', '--always', '--dirty'])\
        .decode('utf-8').strip()

    if 'dirty' in commit and run_name is not None:
        raise RuntimeError("Try to run in dirty commit with run name")

    if not isinstance(args, dict):
        args = vars(args)
    data = {
        **args, 'identifier': identifier,
        'source_commit': commit
    }
    logging.info(pformat(data))
    with open(out_path(filename='config.json'), 'w') as f:
        json.dump(data, f, indent=2)

    tb_writer = SummaryWriter(out_path(category='tb'))

    return tb_writer, out_path


def tensor_read_image(path):
    img = Image.open(path).convert('RGB')
    t = torch.tensor(np.array(img)).permute([2, 0, 1]).float() / 255
    return t

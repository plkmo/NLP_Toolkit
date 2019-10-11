# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:09:24 2019

@author: WT
"""
from nlptoolkit.generation.infer import infer_from_pretrained
from nlptoolkit.utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    outputs = infer_from_pretrained(input_sent=None, tokens_len=100, top_k_beam=1)
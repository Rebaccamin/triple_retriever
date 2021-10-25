import json
import nltk
import logging
import pickle
from tqdm import tqdm
import argparse
import string
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO

                        )

logger = logging.getLogger(__name__)


class Triple_cluster():
    def __init__(self):
        self.name='triple cluster'

    def read(self,filename):
        logger.info('load data from ' + filename)
        if '.json' in filename:
            with open(filename, 'r')as f:
                data = json.load(f)
            return data
        elif '.pkl' in filename:
            with open(filename, 'rb')as f:
                data = pickle.load(f)
            return data
        elif '.txt' in filename:
            with open(filename, 'r')as f:
                data = f.readlines()
            return data
        else:
            print('cannot read data from the file!')
            return
        return

    def sibling(self):
        return

    def parents(self):
        return



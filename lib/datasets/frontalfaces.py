from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Adriano de Araujo Abreu Mourao
# --------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle


class frontalfaces(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'image_set' + image_set)
        dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        self._image_set = image_set

        X_train, X_t, y_t, y_t = train_test_split(
                X, y, test_size=0.25, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
                X_t, y_t, test_size=0.5, random_state=42)

        if image_set == 'train':
            self._image_index = dataset.X_train
            self._classes = y_train
        elif image_set == 'val':
            self._image_index = X_val
            self._classes = y_val
        elif image_set == 'test':
            self._image_index = X_test
            self._classes = y_test

        self._class_to_ind = dict(
                list(zip(self.classes, list(range(self.num_classes)))))
        self.competition_mode(False)

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        return ''

    def _get_widths(self):
      return [self._image_index[i].size[0]
              for i in range(self.num_images)]

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        pass

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()

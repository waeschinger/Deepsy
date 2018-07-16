import numpy as np
import matplotlib as plt
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *

from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "data/dogscats/"
sz = 64

torch.cuda.is_available()

torch.backends.cudnn.enabled

# Architectue, resnext50 läuft länger aber viel besser
arch = resnet34
epochs = 2
learning_rate = 0.01

tfms = tfms_from_model(resnet34, sz)
# ImageClassifier takes the data and their  labels from their respective Paths, arch sets the pretrained
# """ """architecture, sz shows the image size)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)

learn.fit(learning_rate, epochs)



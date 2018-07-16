import numpy as np
import matplotlib as plt
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


class Neuralo():
    def __init__(self):
        self.hello = 'Hell World'


    def greeting(self):
        return print("Was geht aber hier "+ self.hello)



nn = Neuralo()

nn.greeting()

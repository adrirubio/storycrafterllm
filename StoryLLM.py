import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# load the bookcorpus dataset
dataset = load_dataset("bookcorpus")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """ CNN in the character-based convolutional encoder"""
    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size=5):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.num_filters = num_filters
        self.max_word_length = max_word_length
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(char_embed_size, num_filters, kernel_size=kernel_size, bias=True)
        self.max_pool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        x_conv = self.conv(x)
        x_conv_out = self.max_pool(F.relu(x_conv)).squeeze()

        return x_conv_out

### END YOUR CODE


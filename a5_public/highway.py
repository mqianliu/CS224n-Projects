#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """ Highway module in the character-based convolutional encoder"""
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate = nn.Linear(word_embed_size, word_embed_size, bias=True)

    def forward(self, x_conv_out):
        x_proj = F.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_proj))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1 - x_gate), x_conv_out)
        return x_highway

### END YOUR CODE 


if __name__ == '__main__':
    """ Test code """
    h = Highway(100)
    x = torch.zeros(100)
    print(x)
    out_h = h(x)
    print(out_h)
    print(torch.mean(out_h))
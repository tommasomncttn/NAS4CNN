# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================

import torch.nn as nn
import random

# ===========================================
# ||                                       ||
# ||       Section 2: Utils 4 EA           ||
# ||                                       ||
# ===========================================


class Utils4EA():

  def __init__(self):
    pass

  @staticmethod
  def check_validity_genotype(input_list):
    list_lengths = [3, 2, 5, 4, 10]

    if len(input_list) != sum(list_lengths):
        return False

    index = 0
    for length in list_lengths:
        sublist = input_list[index:index+length]
        sampled_values = list(range(length))

        if not all(value in sampled_values for value in sublist):
            return False

        index += length

    return True
    
  @staticmethod
  def generate_random_genotype():
    list_lengths = [3, 2, 5, 4, 10]  # Number of elements in each sublist
    random_list = []

    for length in list_lengths:
        sublist = random.sample(range(length), 1)  # Randomly select an element from the sublist
        random_list.extend(sublist)

    return random_list


  @staticmethod
  def generate_random_phenotype():
    genotype = {
        "conv_filters": random.choice([8, 16, 32]),
        "cnn_architecture": random.choice([
            {"kernel_size": 3, "stride": 1, "padding": 1},
            {"kernel_size": 5, "stride": 1, "padding": 2}
        ]),
        "activation": random.choice([nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Softplus(), nn.ELU()]),
        "pooling": random.choice([
            {"kernel_size": 2, "pooling_type": "Average"},
            {"kernel_size": 2, "pooling_type": "Maximum"},
            {"kernel_size": 1, "pooling_type": "Average"},
            {"kernel_size": 1, "pooling_type": "Maximum"}
        ]),
        "linear_1_neurons": random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    }
    return genotype


  @staticmethod
  def genotype_space():
    genotype_space = {}
    for key, value in Utils4EA().phenotype_space().items():
      value_length = len(value)  # Length of the item
      dimension_space = range(0,value_length)
      genotype_space[key] = dimension_space

    return genotype_space

  @staticmethod
  def genotype_space_bounds():
      sampling_space = []
      for key,value in Utils4EA.genotype_space().items():
        sampling_space.append(value)
      return sampling_space


  @staticmethod
  def phenotype_space():

    possible_config = {
  "conv_filters": [8, 16, 32],
  "cnn_architectures": [
      {"kernel_size": 3, "stride" : 1, "padding" : 1},
    {"kernel_size": 5, "stride" : 1, "padding" : 2}
    ],
  "activation": [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Softplus(), nn.ELU()],
  "pooling": [
      {"kernel_size": 2, "pooling_type": "Average"},
      {"kernel_size": 2, "pooling_type": "Maximum"},
      {"kernel_size": 1, "pooling_type": "Average"},
      {"kernel_size": 1, "pooling_type": "Maximum"}
  ],
  "linear_1_neurons": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    return possible_config
# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================



import numpy as np
import random
from torch.utils.data import DataLoader
from models.multiConfigCNN import MultiConfigCNN
from models.trainingUtils import configure_optimizer, fit_epochs
from data.dataModule import TensorizedDigits
from evolutionaryAlgorithm.utilsEA import Utils4EA


# ===========================================
# ||                                       ||
# ||       Section 2: EA                   ||
# ||                                       ||
# ===========================================


class EA(object):

    def __init__(self, pop_size = 20, lAmbda =0.01, n_epochs = 10):

        # load the constructor parameters as attributes
        self.pop_size = pop_size
        self.lAmbda = lAmbda
        self.max_num_param = 206742
        self.n_epochs = n_epochs

        # define the genotypespace TODO
        self.gene_bounds = Utils4EA.genotype_space_bounds()

        # dataset and dataloader
        self.train_data = TensorizedDigits(mode="train")
        self.val_data = TensorizedDigits(mode="val")
        self.test_data = TensorizedDigits(mode="test")
        self.training_loader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False)

    # -------
        # FOR NN
    # -------

    def evaluate(self, x): # x must be a list of models
        fitness_list = []
        for model in x:

          opt = configure_optimizer(model)
          results = fit_epochs(self.training_loader, self.val_loader, self.test_loader, model = model, optimizer = opt, n_epochs = self.n_epochs, test_model = False, is_silent = True)
          
          num_params = sum(p.numel() for p in model.parameters())
          if num_params > self.max_num_param:
            print(num_params)
            print(model)
            raise ValueError("Compute again, unbeliavable shame")
          fitness = results["val_ce_list"][-1] + self.lAmbda*(num_params/self.max_num_param)
          fitness_list.append(fitness)
        
        return fitness_list


    def config_from_single_genotype(self, genotype):

      # checking genotype validity
      is_valid = Utils4EA.check_validity_genotype(genotype)
      #if not(is_valid):
      #  raise ValueError("Invalid Genotype")
      
      # importing phenotype space
      phenotypespace = Utils4EA.phenotype_space()

      # get the phenotype
      phenotype_config = {
            "conv_filters": phenotypespace["conv_filters"][genotype[0]],
            "cnn_architectures": phenotypespace["cnn_architectures"][genotype[1]],
            "activation": phenotypespace["activation"][genotype[2]],
            "pooling": phenotypespace["pooling"][genotype[3]],
            "linear_1_neurons": phenotypespace["linear_1_neurons"][genotype[4]]
                  }
      return phenotype_config


    def encoding_population(self, genotypes):
      phenotypes = []
      for genotype in genotypes:
        phenotype_config = self.config_from_single_genotype(genotype)
        phenotype = MultiConfigCNN(phenotype_config)
        phenotypes.append(phenotype)
      return phenotypes
        
  
    
    # -------
        # FOR EVOLUTIONARY ALGORITHM
    # -------

    def recombination(self, x_parents, f_parents):
      '''first step in working out the candidate solutions, we pass from the 
      selected parents, to a set of INITIAL candidate solutions
      '''
      # initialize an empty list where to insert recombined parents
      x_recombined_parents = [] 

      # loop for half of the population size
      for i in range(self.pop_size//2):

        # pick two random index
        idx = np.random.choice(len(x_parents), size=2, replace=False)

        # get the corresponding elements from the array
        pair = (x_parents[idx[0]], x_parents[idx[1]])
        
        # select a random point on the lists
        index = random.randrange(len(pair[0]))

        # recombine them with one point crossover
        if index != 0:
          recombined_parent_1 = [*pair[0][:index], *pair[1][index:]]
          recombined_parent_2 = [*pair[1][:index], *pair[0][index:]]
        else: 
          recombined_parent_1 = [pair[0][0], *pair[1][1:]]
          recombined_parent_2 = [pair[1][0], *pair[0][1:]]

        # append to the new list
        x_recombined_parents.append(recombined_parent_1)
        x_recombined_parents.append(recombined_parent_2)

      # convert list to numpy array
      x_recombined_parents_array = np.stack(x_recombined_parents)

      # return the array
      return x_recombined_parents_array
  
        

    def mutation(self, x_children):

      # initialized mutated_children
      mutated_children = []

      # repeat for each solution in the set of candidate solutions
      for x_current in x_children:

        index = random.randrange(len(x_current))
        sampling_range = self.gene_bounds[index]
        random_mutation = random.choice(sampling_range)
        x_current[index] = random_mutation
        mutated_children.append(x_current)

      # Convert the list to a NumPy array of arrays
      mutated_children_array = np.stack(mutated_children)

      # return the final candidate solutions
      return mutated_children_array


    def survivor_selection(self, x_parents_geno, x_children_geno, f_parents, f_children):
      '''we pass from a set of FINAL candidate solution, to the new population
      by selecting the best individuals
      '''

      # concatenate miu and lambda, old population and candidate solution
      x = np.concatenate([x_parents_geno, x_children_geno])

      # concate their fitness
      f = np.concatenate([f_parents, f_children])

      # sort them, the lefter the better
      index = np.argsort(f)
      x = x[index]
      f = f[index]

      # return only the best pop size (miu)
      return x[:self.pop_size], f[:self.pop_size]

    # -------
        # STEP
    # -------

    def step(self, x_parents_geno, f_parents):

        # recombination step 
        x_children_geno = self.recombination(x_parents_geno, f_parents)
        
        # mutation step
        x_children_geno = self.mutation(x_children_geno)

        # encoding step 
        x_children_pheno = self.encoding_population(x_children_geno)

        # evaluation step 
        f_children = self.evaluate(x_children_pheno)

        x, f = self.survivor_selection(x_parents_geno, x_children_geno, f_parents, f_children)

        return x, f
# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================


import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torchview import draw_graph

from models.multiConfigCNN import MultiConfigCNN
from models.trainingUtils import configure_optimizer, fit_epochs, plot_results
from evolutionaryAlgorithm.EA import EA
from evolutionaryAlgorithm.utilsEA import Utils4EA


# ===========================================
# ||                                       ||
# ||       Section 2: Optimizer For EA     ||
# ||                                       ||
# ===========================================

class Optimizer:

  def __init__(self,generation = 50, pop_size = 20, n_epochs_generation = 10, lAmbda =0.01, epochs_best_model = 10):

    self.generation = generation
    self.pop_size = pop_size 
    self.n_epochs_generation = n_epochs_generation
    self.lAmbda = lAmbda
    self.epochs_best_model = epochs_best_model
    self.ea = EA(pop_size = self.pop_size, n_epochs = self.n_epochs_generation, lAmbda =self.lAmbda)
  
  def n_generation(self):
    
    # init genotype population
    x_geno = []
    for i in range(0,self.pop_size):
      genotype = Utils4EA.generate_random_genotype()
      x_geno.append(genotype)

    # getting phenotype and objective function
    x_pheno = self.ea.encoding_population(x_geno)
    f = self.ea.evaluate(x_pheno)

    num_generations = self.generation  
    pop_size = self.pop_size

    f_best = [np.array(f).min()]
    f_hisotry = [np.array(f).min()]
    x_best = x_geno[np.array(f).argmin()]
    visual_range = tqdm(range(num_generations))
    for i in visual_range:
      print("Generation: {}, best objective: {:.5f}".format(i, np.array(f).min()))
      x_geno, f = self.ea.step(x_geno, f)
      f_hisotry.append(np.array(f).min())
      if np.array(f).min() < f_best[-1]:
          f_best.append(np.array(f).min())
          x_best = x_geno[np.array(f).argmin()]
      else:
          f_best.append(f_best[-1])
      visual_range.set_postfix({f'best objective score at generation {i} ': f_best[-1]})
    visual_range.close()
    print("FINISHED!")
    return f_best, x_best, f_hisotry

  def printout_best_genotype(self, x_best):

    genotype = x_best
    phenotype_config = self.ea.config_from_single_genotype(genotype)
    phenotype = MultiConfigCNN(phenotype_config)
    input = next(iter(self.ea.training_loader))[0]
    model_graph = draw_graph(phenotype, input_data=input)
    print("\033[31mBEST GENOTYPE :", "\033[0m")
    print("")
    print("==>",x_best)
    print("")
    print("")
    print("\033[31mITS PHENOTYPE CONFIGURATION:","\033[0m")
    print("")
    print("==>",phenotype_config)
    print("")
    print("")
    print("\033[31mBEST VISUAL OF THE MODEL", "\033[0m")
    print("")

    return model_graph.visual_graph

  def plot_obj(self, f_best, f_history):

    generations = range(1, len(f_best) + 1)

    plt.plot(generations, f_best, label='Global Best')
    plt.plot(generations, f_history, label='Local Best')

    plt.xlabel('Generation')
    plt.ylabel('Objective')
    plt.title('Objective Progression')
    plt.legend()
    plt.grid(True)

    plt.show()
  
  def train_and_test_selected_genotype(self, x_best):

    genotype = x_best
    phenotype_config = self.ea.config_from_single_genotype(genotype)
    phenotype = MultiConfigCNN(phenotype_config)
    opt = configure_optimizer(phenotype)
    results = fit_epochs(self.ea.training_loader, self.ea.val_loader, self.ea.test_loader, model = phenotype, optimizer = opt, n_epochs = self.epochs_best_model)

    return results

  def print_results_of_model(self, results):

    print("TEST CE: ", results["test_ce_result"])
    print("TEST LOSS: ", results["test_loss_result"])

    plot_results(results)

  def analysis(self):
    print("")
    print("")
    print("\033[2J\033[\033[1m" + "Analysis 1: generations" + "\033[0m")
    print("")
    print("")
    f_best, x_best, f_hisotry = self.n_generation()
    print("")
    print("")
    print("\033[2J\033[\033[1m" + "Analysis 2: results of generation in terms of objective" + "\033[0m")
    print("")
    print("")
    self.plot_obj(f_best, f_hisotry)
    print("")
    print("")
    print("\033[2J\033[\033[1m" + "Analysis 3: training and testing best genotype" + "\033[0m")
    print("")
    print("")
    results = self.train_and_test_selected_genotype(x_best)
    print("")
    print("")
    print("\033[2J\033[\033[1m" + "Analysis 4: results of best phenotype in terms of loss and classification error" + "\033[0m")
    print("")
    print("")
    self.print_results_of_model(results)
    print("")
    print("")
    print("\033[2J\033[\033[1m" + "Analysis 5: printing out best genotype, phenotype configuration, and phenotype" + "\033[0m")
    print("")
    print("")
    return self.printout_best_genotype(x_best)
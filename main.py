import sys
from evolutionaryAlgorithm.optimizerEA import Optimizer

def main(generation, pop_size, n_epochs_generation, lAmbda, epochs_best_model):
    ot = Optimizer(
        generation=generation,
        pop_size=pop_size,
        n_epochs_generation=n_epochs_generation,
        lAmbda=lAmbda,
        epochs_best_model=epochs_best_model
    )
    ot.analysis()

if __name__ == "__main__":
    # Get command-line arguments
    args = sys.argv[1:]  # Exclude the script name itself

    # Check if all required arguments are provided
    if len(args) != 5:
        print("Usage: python main.py generation pop_size n_epochs_generation lAmbda epochs_best_model")
        sys.exit(1)

    # Parse command-line arguments
    generation = int(args[0])
    pop_size = int(args[1])
    n_epochs_generation = int(args[2])
    lAmbda = float(args[3])
    epochs_best_model = int(args[4])

    # Call the main function with the parsed arguments
    main(generation, pop_size, n_epochs_generation, lAmbda, epochs_best_model)

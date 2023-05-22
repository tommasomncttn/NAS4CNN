# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================


import matplotlib.pyplot as plt
import torch


# ===========================================
# ||                                       ||
# ||       Section 2: Optimizer            ||
# ||                                       ||
# ===========================================

def configure_optimizer(model, lr = 1e-3, wd = 1e-5):
  return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)

# ===========================================
# ||                                       ||
# ||       Section 3: trainig              ||
# ||                                       ||
# ===========================================

def train_one_epoch(dataloader, model, optimizer, epoch_n = None, is_silent = False):

    model.train()

    size = len(dataloader.dataset)
    total_loss = 0
    total_miss = 0
    train_step = 0

    for (X, y) in dataloader:

        train_step += 1

        # logits
        log_prob = model(X)

        # classification
        predictions = model.classify(log_prob)

        # misclassified 
        missclassified = model.count_misclassified(predictions, y)
        total_miss += missclassified

        # loss
        loss = model.compute_loss(log_prob, y, reduction = "sum")
        total_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    
    # compute epoch statistics
    avg_loss = total_loss / size
    avg_ce = total_miss / size
    
    if not(is_silent):
      print("")
      print(f" TRAINING => Results of epoch number {epoch_n}:")
      print('')
      print('    Average training loss: {0:.5f}'.format(avg_loss))
      print('    Average training classification error: {0:.5f}'.format(avg_ce))

    return avg_loss.item(), avg_ce

# ===========================================
# ||                                       ||
# ||       Section 3: validation/test      ||
# ||                                       ||
# ===========================================

def evaluation_one_epoch(dataloader, model, optimizer, mode = "validation", epoch_n = None, is_silent = False):

    model.eval()

    size = len(dataloader.dataset)
    total_loss = 0
    total_miss = 0
    val_step = 0


    with torch.no_grad():

      for (X, y) in dataloader:

          val_step += 1

          # logits
          log_prob = model(X)

          # classification
          predictions = model.classify(log_prob)

          # misclassified 
          missclassified = model.count_misclassified(predictions, y)
          total_miss += missclassified

          # loss
          loss = model.compute_loss(log_prob, y, reduction = "sum")
          total_loss += loss



    # compute epoch statistics
    avg_loss = total_loss / size
    avg_ce = total_miss / size


    if not(is_silent):
      if mode == "validation":

        
        print('')
        print('    Average validation loss: {:.5f}'.format(avg_loss))
        print('    Average validation classification error: {:.5f}'.format(avg_ce))


      else:

    
        print("")
        print(f"TESTING => Results of over test set")
        print('')
        print('    Average test loss: {:.5f}'.format(avg_loss))
        print('    Average test classification error: {0:.5f}'.format(avg_ce))


    return avg_loss.item(), avg_ce


# ===========================================
# ||                                       ||
# ||       Section 4: trainer function     ||
# ||                                       ||
# ===========================================


def fit_epochs(training_loader, val_loader, test_loader, model, optimizer, n_epochs, test_model = True, is_silent = False):

  train_loss_list = []
  train_ce_list = []
  val_loss_list = []
  val_ce_list = []
  test_loss_result = []
  test_ce_result = []

  for i in range(0,n_epochs):

    train_loss, train_ce =  train_one_epoch(dataloader = training_loader, model = model, optimizer = optimizer, epoch_n = i, is_silent = is_silent)
    train_loss_list.append(train_loss)
    train_ce_list.append(train_ce)

    val_loss, val_ce = evaluation_one_epoch(dataloader = val_loader, model = model, optimizer = optimizer, epoch_n = i, is_silent = is_silent)
    val_loss_list.append(val_loss)
    val_ce_list.append(val_ce)

  if test_model:
    test_loss, test_ce = evaluation_one_epoch(dataloader = test_loader, model = model, optimizer = optimizer, mode = "test", epoch_n = i, is_silent = is_silent)
    test_loss_result.append(test_loss)
    test_ce_result.append(test_ce)


    result_dic = {"train_loss_list": train_loss_list, "train_ce_list": train_ce_list,"val_loss_list":val_loss_list, "val_ce_list":val_ce_list, "test_loss_result":test_loss_result, "test_ce_result":test_ce_result, "number_of_epochs" : n_epochs}
  else: 
    result_dic = {"train_loss_list": train_loss_list, "train_ce_list": train_ce_list,"val_loss_list":val_loss_list, "val_ce_list":val_ce_list, "number_of_epochs" : n_epochs}
  return result_dic


# ===========================================
# ||                                       ||
# ||       Section 4: Visualization        ||
# ||                                       ||
# ===========================================


def plot_results(results):
    # Extracting the data from the dictionary
    train_loss_list = results["train_loss_list"]
    train_ce_list = results["train_ce_list"]
    val_loss_list = results["val_loss_list"]
    val_ce_list = results["val_ce_list"]
    test_loss_result = results["test_loss_result"]
    test_ce_result = results["test_ce_result"]
    n_epochs = results["number_of_epochs"]

    # Plotting losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_loss_list, label="Train Loss")
    plt.plot(range(1, n_epochs + 1), val_loss_list, label="Validation Loss")
    plt.plot(range(1, n_epochs + 1), [test_loss_result] * n_epochs, label="Test Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Losses")
    plt.legend()

    # Plotting classification errors
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), train_ce_list, label="Train Classification Error")
    plt.plot(range(1, n_epochs + 1), val_ce_list, label="Validation Classification Error")
    plt.plot(range(1, n_epochs + 1), [test_ce_result] * n_epochs, label="Test Classification Error", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Classification Error")
    plt.title("Training and Evaluation Classification Errors")
    plt.legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()


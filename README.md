# Neural Architecture Search For Image Classification
The objective of this project is to develop an efficient NAS method that automates the process of designing neural network architectures while achieving high performance on classification tasks. The implementation is focused on a specific type of convolutional neural network (CNN) tailored for image classification. The model space consists of a finite set of possible neural configurations which span from the following structure:

        Conv2d → f(.) → Pooling → Flatten → Linear 1 → f(.) → Linear 2 → Softmax

Each neural network in the optimization process serves as a probabilistic classifier. The training objective is to minimize the negative log-likelihood loss, which is computed as the negative logarithm of the predicted probability of the true class for each data point, summed over all data points. This loss function captures the confidence of the model in assigning the correct labels to the data points.

$$L_{\text{nn}}(\mathbf{y},\mathbf{\hat{y}}) = - \sum_{i=1}^{N} \log(p(y_i | x_i, \Theta))$$

Here, $\Theta$ represents the parameters of the classifier, $N$ is the total number of points in the dataset, $x_i$ and $y_i$ denote the ith datapoint and its true label, and $\hat{y}$ and $\textbf{y}$ correspond to the prediction and true label over the batch.

The genetic algorithm optimization problem aims to find the best neural network architecture. The objective function, denoted as $O(\cdot)$, is a composite measure that balances classification performance on the validation set and model complexity. It is computed as the sum of the classification error on the validation set and a regularization term ($\lambda$) multiplied by the ratio of the current neural network's parameter count ($N_p$) over the maximum parameter count ($N_{\text{max}}$).

<p align="center">
  <img src="https://github.com/tommasomncttn/NAS4CNN/assets/91601166/8911a118-0568-4787-9f93-54fa6916272c" width="370" alt="Image">
</p>




## Evolutionary Algorithm PseudoCode
<p align="center">
  <img src="https://github.com/tommasomncttn/NAS4CNN/assets/91601166/b9272c80-6a3f-48f7-be3e-52599a26ea45" width="550" alt="Image">
</p>



## Results

The EA algorithm has run for 50 generations and selected the best architecture. This architecture is shown in the table below and it is characterized by the following hyperparameters: 32 output channel for the CNN, a kernel of size 3 with a stride of 1 and padding of 1; the activation used is ReLU, while the pooling type is Maximum; lastly, the first linear layer has 70 neurons. In terms of performance, this architecture reaches a loss of 0.1930 and a classification error of 0.0604.
<p align="center">
  <img src="https://github.com/tommasomncttn/NAS4CNN/assets/91601166/dbf4411e-1885-444a-b7dd-4471a6032855" width="200" alt="Image">
</p>



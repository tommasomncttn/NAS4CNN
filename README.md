# Neural Architecture Search For Image Classification
The objective of this project is to develop an efficient NAS method that automates the process of designing neural network architectures while achieving high performance on classification tasks. The implementation is focused on a specific type of convolutional neural network (CNN) tailored for image classification. The model space consists of a finite set of possible neural configurations which span from the following structure:

        Conv2d → f(.) → Pooling → Flatten → Linear 1 → f(.) → Linear 2 → Softmax

Each neural network in the optimization process serves as a probabilistic classifier. The training objective is to minimize the negative log-likelihood loss, which is computed as the negative logarithm of the predicted probability of the true class for each data point, summed over all data points. This loss function captures the confidence of the model in assigning the correct labels to the data points.

$$L_{\text{nn}}(\mathbf{y},\mathbf{\hat{y}}) = - \sum_{i=1}^{N} \log(p(y_i | x_i, \Theta))$$

Here, $\Theta$ represents the parameters of the classifier, $N$ is the total number of points in the dataset, $x_i$ and $y_i$ denote the ith datapoint and its true label, and $\hat{y}$ and $\textbf{y}$ correspond to the prediction and true label over the batch.

The genetic algorithm optimization problem aims to find the best neural network architecture. The objective function, denoted as $O(\cdot)$, is a composite measure that balances classification performance on the validation set and model complexity. It is computed as the sum of the classification error on the validation set and a regularization term ($\lambda$) multiplied by the ratio of the current neural network's parameter count ($N_p$) over the maximum parameter count ($N_{\text{max}}$).

<p align="center">
  <img src="https://github.com/tommasomncttn/NAS4CNN/assets/91601166/09b367bd-9add-425f-8334-afa3f3c57f38" width="370" alt="Image">
</p>



## Evolutionary Algorithm PseudoCode
![image](https://github.com/tommasomncttn/NAS4CNN/assets/91601166/28a941cc-208b-424f-af4c-5460b60f360c)



# Reinforcement Learning
Started with reading Sutton and Bartos Intro book while following David Silvers class. Onto more book and resources!

**ActorCritic.py:** Implements Actor Critic algorithm. Tested on BipedalWalker from OpenAI Gym.

**DeepQ_RubiksCube.zip:** Contains a few files for attempting to solve a Rubiks Cube with Deep Q learning. More for practice with writing reward functions and implementing the Deep Q-learning algorithm. Never actually solved the cube, but was clearly learning to try and maximize the amount of correct placed spaces. Saw states such as having 2 sides completely done with the others 2 off ect. To actually be able to solve it, I suspect adding some heuristics to guide the agent/cut down state space, and a better suited reward function would be necessary. But again, just for general practice. 

**GPI_NN.py:** Implemented a General Policy Iteration using 2 Neural Networks. One representing the Policy, and the other representing the Action Value Function. Adapted another gradient descent step for nudging towards improved actions during the training of the Policy Network. Works for continuous action spaces. 

**MCUpdated_CartPoly.py:** Classic Monte Carlo method for solving the CartPole problem. Broke the states into descrete values, and used a look-up table. 

**PolicyGradient.py:** Vanilla Policy Gradient method (tested on CartPole). Added a custom "biased experience replay" buffer, which stores states that had the highest probablity for their episode. Additionally trains on this set of data every so often. Of course, the probablity for those states may not be the same by the time we are trained on them, then from when we added them to the buffer. So if functions both as a reinforcer for convergence (if they are), and helps to undue catastrophic forgetting (if they aren't).

**PyTorchTraining.ipynb:** Some exercises for getting used to Pytorch, which I use for any Reinforcement Learning projects.

**TD_MountainCar.py:** Implemented TD algorithm. Pretty bare bones, actually no eligiblity traces. Uses discretized state space, and look-up table. Did solve it though! 

# MachineLearning
Machine Learning class using Google Colab

**BlurrImage.py:** Passes an NxN mean filter over the image to blurr it. Uses 2 passes of matrix multiplication to do so.

**Ensemble.jl:** Applies Evolutionary Algorithms to the task of classification in Machine Learning. Keeps a population of each class in the data, and iteratively evolves each population towards the target data to be classified. Has no formal training phase, instead requiring the algorithm to run for each new classification. Whichever population evolves and converges to the targets attributes/features fastest, is considered to have been the most similar and therefore the proper labeling for this data. Can be done in ensemble, although the slower computation time of this algorithm isn't suitable for large ensembles. May be well suited for smaller data-sets, without a large amount of features. Seems to be relatively robust and insensitive to smaller "training" data. Tested against a naive implementation, which just computes an average "similarity score", between all populations and the target data initially. Using the same formula that the Evolutionary implementation uses for fitness and convergence evaluation. I have only tested on a select few data sets, although my findings indicate that the Evolutionary Algorithm is superior to the naive implementation, and on par with using a RandomForestClassier.  

## Tutorials (following them to learn!)
**RandomForest.jl:** Implementing a RandomForest to predict tomorrow's weather based on a year of data for the city. Used Julia, to become familiar with their wrapper libraries for Scikit-learn, and practice with DataFrames.jl. Learned about the robustness of RandomForests to noise in the data, and how feature scaling isn't required for their efficient performance (being an ensemble of Deicsion Trees) 

## IntroProject 
### My final project from my Intro to Machine Learning class. Link for Kaggle dataset I used below.
Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

**AddingPointsRegression.ipynb:** A custom attempt at tweaking Polynomial Regression. Tries to add new instances to the data (all features, and a label), by interpolating points into the dataset. Uses a recursive “dimensional average breakdown”, to try and add these points. Tested for smaller visualizable dimensions graphically, and the logic should apply to higher dimensions, perhaps with less accuracy (curse of dimensionality). The hope is that these added points will be more or less centered around a desirable curve of best fit, to help facilitate said curve being drawn. Seems to neither hurt or improve accuracy significantly. From testing on 2D, the less of these points there are, the more centered around the desired “best fit” area they are. Deeper (recursively) generation of points offers less centered points. This behavior was expected, which is why I implemented a min_samples variable to the function to help limit the depth and splitting of subspaces. The goal is to add a small amount of meaningful points to help guide the training. This tradeoff is likely the reason no substantial change (good or bad), is seen in the accuracy. May be more useful for smaller less dimensional data sets though.

**FuzzyRegression.ipynb:**  Another custom attempt at tweaking Polynomial Regressors. The motivation was to take many strong learners (Polynomial Regressors fitted to data without regularization), and generalize them to make them weaker. Initially each learner was trained on a non-overlapping subset of the data (call this S_i). Then for subsequent training, the entire training set was used. Although the labels for each learner were different. Every learner received the correct labels for their data from S_i, but received “fuzzy labels” for the rest of their data. Supplied by the collection of every other learner predicting values for their S_i. This was really entirely experimental. I wanted to see what would happen if the labels for every learning changed over time. The idea was that initially these fuzzy labels would be accurate to the true labels (since each learner is initially overfitting to their S_i). Over time, each learner would generalize and become weaker. It is assumed each learner would become less accurate on data in their S_i, but more accurate for all other points. This would result in the fuzzy labels becoming “fuzzier and fuzzier” as time went on. I thought maybe perhaps this would cause the learners to generalize more, since they are potentially being trained on fuzzier data. Of course data that is so inaccurate is not beneficial. That’s why I was imposing early stopping after every iteration of running gradient descent for a few times with a low learning rate. Once the average of all learners was less than a previous run, stop the algorithm and return the learners. The results are in the notebook. I show the results of each learner initially, and after weakening. Since each learner is initially trained on a smaller subset of data, the increase in accuracy surely could be solely from the learners being exposed to more data after weakening. It’s hard to say what influence the method has with only these results. No further testing was done.

**InitialAttempt.ipynb:** First attempt at cleaning data and implementing Machine Learning algorithms. I experimented with the following estimators: Linear, Polynomial, and Ridge Regression. Random Forest and Gradient Boosting. Neural Networks.

**SecondAttempt.ipynb:** Second attempt. The data cleaning process was improved, through learned use of Imputer's and Scalers. Use Scipy optimizers to attempt and find optimal parameters for the Neural Network (Neurons per layer, activation function combination ect). Overall these two attempts at Machine Learning for the first time were extremely informative.

## Unscrambler
### Computer Vision application for unscrambling an image shuffled by 2x2 blocks (the 4 corners)

**ScrambledPets.ipynb:** Originally training and processing the dataset. I used the Oxford-IIIT Pets Dataset, which contains images of cats and dogs. Fastai, a high level deep learning library for Python was used as well. My approach was to train the model (resnet34, pretrained) on scrambled images, where the label could be mapped to a permutation key for how to unscramble it. 

**ProjectDemo.ipynb:** Demonstrates the proficiency of the above model on 3 exclusive test sets. Dogs, Cats, and Miscellaneous which has random animals the model was not trained on. It achieves an average accuracy of about 95 across all sets. 

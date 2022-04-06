![imageMain](https://user-images.githubusercontent.com/54413900/161870408-8ca8fe9f-bbaf-480c-8fa8-bd9c3b63eba1.png)




## Introduction / Background:

The National Basketball Association (NBA) consists of 30 teams, each containing 15 players, with additional players rotating in during the playoffs. Near the end of each season, a three-day exhibition event known as NBA All-Star Weekend is held, pitting the biggest stars across the league against each other. The All-Star lineup is a fan, player, and media voted list of the NBA’s 24 biggest players for the current season based on their playing position. Our project uses a dataset on players from the past 20 NBA seasons to best estimate which players would end up on the NBA All-Star lineup for the 2021-2022 season.

## Problem Definition:

This prediction is not as straightforward as it may initially seem. A huge amount of raw data on NBA player statistics and performance is recorded and made publicly available each season, but it is not always immediately apparent how these statistics translate to perceptions off-court in what is in some ways a popularity contest. By applying machine learning techniques, however, we plan to make accurate predictions for a seemingly unscientific process.






## Data Collection:

###### Dataset Information/Source:

The website Basketball-Reference.com contains a wide variety of past and current player information. For our investigation we collected a dataset of advanced player statistics, which contained 23 features and 9493 data points. Some of the features included player_efficiency_rating, usage_percentage, and win_shares. A Python script was used to scrape the data from the past 20 seasons and assemble in preliminary .csv files. Manual cleaning was then undertaken to ready the datasets for further manipulation.

###### Preprocessing:

Initial investigations of the cleaned data revealed somewhat chaotic results, which is perhaps not surprising given the large number of features. We generated an initial heat map to see how features were correlated. This heat map is shown below:

![Image2Main](https://user-images.githubusercontent.com/54413900/161870467-bf4a5f04-cf20-405f-8b76-0a48a54fce53.png)


This tells us there are a number of features which are highly correlated, meaning the model really only needs one of them to function well. We would thus want to consider eliminating one of those features highlighted in dark green or red for feature reduction purposes.

We wanted to understand more about the features that were important to the model, so we then ran a Random Forest Classifier on the full set of features. This clarified a number of insights, highlighting a number of features that seemed to be highly relevant, especially value_over_replacement_player, win_shares, player_efficiency_rating, and usage_percentage. In this graph, features with larger bars indicate higher ranking/importance to the model.

![Image4](https://user-images.githubusercontent.com/54413900/161873108-939b2954-1e7c-48be-80b1-955559dac669.png)


We wanted to confirm these findings via another selection strategy, and thus ran a Ridge Classifier on the same data.

![image5](https://user-images.githubusercontent.com/54413900/161873136-630f413e-9622-4144-a49e-29ce7debf99c.png)

As seen, a more complicated picture emerged. True_shooting_percentage was identified as an extremely important feature, along with win_shares_per_48_minutes and value_over_replacement. 


Since a feature selection trend was not necessarily clear, we decided to continue by running Forward and Backward Selection using different estimators, each time selecting the top 8 features. These graphs indicate rankings of features, with smaller bars indicating higher ranking/feature importance to the model.

![image6](https://user-images.githubusercontent.com/54413900/161873159-f99c1a81-9f88-4560-b7bd-3f6d72f0c3de.png)

![image7](https://user-images.githubusercontent.com/54413900/161873196-defd2e11-e47d-4b96-868e-24b3e4d83cda.png)

The prediction results agree with some of the earlier findings, giving us confidence to begin finally eliminating some of the features. After discussion, a condensed selection of 9 features was established. These were:

* value_over_replacement_player
* true_shooting_percentage
* win_shares_per_48_minutes
* player_efficiency_rating
* usage_percentage
* offensive_box_plus_minus
* three_point_attempt_rate
* free_throw_attempt_rate
* steal_percentage

Additionally, we realized we needed to standardize player features by position and year, since the context of the era of basketball and the position played heavily dictate the outcome of the all-star lineup. As an example, a power forward would tend to have a higher three_point_attempt_rate than a center in the NBA by the role of their position on the court. Additionally, since 2015, three_point_attempt_rate has increased across all positions compared to a season such as 2003.

## Methods:

### First Model: Binary Classifier

Our first model consisted of a neural net binary classifier implemented in PyTorch. Given a datapoint of single player's stats over one season, it will predict whether it belongs to either the 'All-Stars' or 'Non-All-Stars' class. Below are the key details of our model architecture and data:
* **Number of Hidden Layers: 3**
    * Input Layer: Shape of input data
    * Layer 1: 50 neurons
    * Layer 2: 25 neurons
    * Layer 3: 10 neurons
    * Output Layer: 1 neuron
* **Activation Function:**
    * ReLU for every layer except output
    * Sigmoid for output layer
* **Loss Function:** Binary Cross Entropy
* **Optimizer:** Stochastic Gradient Descent
* **Train/Test/Validation Split:** 60%/20%/20% 

Because the datasets inherently contain far more non All-Stars than All-Stars, there is a significant imbalance between the classes. For instance, only about 5% of players in the last 20 seasons of the NBA were selected as All-Stars. Thus, optimizing our model for solely raw accuracy did not give us a classifier with much predictive power, as it could predict 'Non-All-Star' for any datapoint and still achieve a ~95% overall accuracy. We attempted to solve this through two methods: Optimizing for the F1-Score of the 'All-Stars' class and oversampling. Oversampling refers to the technique of randomly sampling datapoints belonging to the minority class such that both classes have the same number of datapoints in the training set, thus creating a more balanced dataset.

## Results and Discussion: 

We manually experimented with different neural network architectures and found that the architecture in the previous section performed the best for the classification task. Additionally, we use ReLU as the activation function for all layers except for the output layer because it is the state-of-the-art default. We use sigmoid after the output layer to convert the output to a probability between 0 and 1. This probability can then be rounded to either 0 or 1 corresponding to the labels of 'Non-All-Star' and 'All-Star' respectively for inference. Also, we used binary cross entrophy and stochastic gradient descent as they are defaults for binary classification.

We use a three way split of data into train, validation, and test sets so that we could use the train set to train the neural network, the validation set to select the best model and hyperparameters, and the test set to calculate the final performance metrics.

First, we tried training and evaluating our classifier with the full set of standardized features (23 features from Basketball-Reference.com). We found that our model was able to achieve 97.36% raw accruacy and a 0.77 F1-score on the 'All-Star' class with the below hyperparameters.

**Optimized Hyperparameters:**
* Learning rate: 0.05
* Number of epochs: 14
* Batch Size: 32

At this stage, we manually optimized the hyperparameters for F1 of the 'All-Star' class, however, in the the future, we could use hyperparameter tuning to optimize these in an automated manner. The full classification report from sklearn for this full standardized dataset is provided below:

<img width="448" alt="Screen_Shot_2022-04-06_at_1 50 24_AM" src="https://user-images.githubusercontent.com/37726288/161912908-63791fff-2ac0-43d0-8d79-6987a1acb5af.png">

The graph of loss for the train and validation set for each epoch is shown below:

![loss_v_epoch](https://user-images.githubusercontent.com/37726288/161913210-97b441df-8513-4e9f-a76e-d33fcb7fed43.png)

Since the loss on the validation set does not begin to increase in the epoch range shown, we can be confident that the model is not overfitting to the training set.
Also, the graph of accuracy for the train and validation set for each epoch is shown below:

![acc_v_epoch](https://user-images.githubusercontent.com/37726288/161913288-85fabb7f-92de-4add-b026-c0175eec2432.png)

We also tried evaluating our classifier with the reduced set of 9 standardized features mentioned above. However, we found that for the chosen model architecture, the model trained on these reduced features performed worse than the model trained on the full set of standardized features as this model had a 95.45% raw-accuracy and a F1-score of 0.62 on the 'All-Star' class. 

Finally, we found that applying PCA applied to both the full and reduced feature sets give much worse accuracy and F1-scores for this type of model architecture.

In the future, we hope to experiment with different models such as Random Forest or Naive Bayes models to see if the reduced feature set and the application of PCA will perform better with these architectures. Also, as mentioned, we hope to perform rigorous hyperparameter tuning in order to find a truly optimal model for predicting the All-Star team.

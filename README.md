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

These would be the final features used in our machine learning models.

## Methods:

###### First Model:

Our first model used PyTorch to generate a neural network using three layers. Because the datasets inherently contain many more non All-Stars than All-Stars, there is a significant imbalance between the classes, thus calculating raw accuracy did not give us a good sense of what was going on when initially applying our neural network model. We attempted to solve this through two methods: F1 score adjustment and oversampling the minority class (the All-Stars). Both cases saw improvements, with F1 Score adjustments pushing the accuracy of the model to 91% (for Non-All stars) / 61% (for All-Stars). While oversampling the result still shows the same test case.

Additionally, we realized we needed to normalize players by position and year, since different positions inherently put up different types of statistics and to account for changes over the past two decades in the way the sport is played. As an example, a power forward would tend to have more three_point_attempt_Rate than Center by the nature of their position and roll on the court. We re-ran the model using the normalized Z-scores, observing a 2% increase in accuracy. 

## Results and Discussion: 

After a tedious process of testing multiple results and data sets. We realized that putting in PCA heavily skewers the data. We then reduced the data set from the original 23 features to 9 important features along with normalizing it. The 99% accuracy for Non-All Stars and the 61% for All-Stars is shown to be accurate. In every essence of the word, given that a player is Non-All Star, our model would have a 99% accuracy. While for a All-Star our model would have a 61% accuracy of predicting correctly. 

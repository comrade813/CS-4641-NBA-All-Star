### Overview
Our project uses a dataset on players from the past 20 NBA seasons to best estimate which players would end up on the NBA All-Star lineup for the 2021-2022 season. The All-Star lineup is a fan, player, and media voted list of the NBA’s best players for the current season based on their playing position.

Previous work has shown success in predicting various NBA accolades using machine learning techniques. For example, Ji and Li use various neural network architectures as well as adaptive boosting with a dataset of in season player statistics and weekly fan voting to predict whether a player would be an NBA all star starter or reserve [1]. Li used methods such as random forests, decision trees, and logistic regression to predict an NBA MVP out of likely candidates [2]. Also, Nguyen et al. utilize historical player data to forecast a player’s future performance and popularity as characterized by all-star selection using techniques such as linear regression and support vector machine [3].

We use a dataset from Basketball-Reference.com contains information ranging from player positions, age, associated teams, games played/started, minutes per game, field goals made, field goals attempted, field goals percentage, 2 and 3 point percentage, total rebounds, assists, steals, blocks, turnovers, personal fouls, total points made and some advanced statistics such as eFG% and PER. For more information, please visit the [basketball-reference](https://www.basketball-reference.com/leagues/NBA_2021_totals.html) site.
to see the players stats for just the 2020-2021 season. 

### Problem Definition
Many metrics exist that analyze on-court player performance, but relatively few connections exist for translating these statistics to off-court perceptions and assessments. Given player statistics for a particular season, we seek to develop a method for accurately predicting a key annual NBA selection that is based on audience, player, and coach perceptions - the All-Star player lineup. This consists of the 24 NBA players who are regarded by coaches, players, and fans as the best players at their positions across the league. 

### Methods
Because predicting if a player will be an NBA All-Star is highly dependent on the player’s in-season and historical statistics, supervised learning, specifically classification, is required. We choose to use a neural network along with a random forest model because previous work by Ji and Li, Li, Nguyen et al. found that these types of models work well to predict basketball award selections such as the MVP (most valuable player) or the allstars.

### Potential Results
We plan to test our models on the last 20 seasons of NBA data and associated all-star teams. We can then calculate classification metrics such as accuracy, precision, recall, and the F1-score. The latter will be useful because our data is inherently not balanced (since we have many times more non-All-Stars than All-Stars). Additionally, we plan to calculate the percentage of All-Star players that our model was able to predict for every year in the test set. Finally, we plan to include a discussion comparing the performance of our models with existing work.

### References
[1] B. Ji and J. Li, "NBA All-Star Lineup Prediction Based on Neural Networks," 2013 International Conference on Information Science and Cloud Computing Companion, 2013, pp. 864-869, doi: 10.1109/ISCC-C.2013.92.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6973701

[2] Li, X. (2021, December). National Basketball Association Most Valuable Player prediction based on machine learning methods. In Second IYSF Academic Symposium on Artificial Intelligence and Computer Engineering (Vol. 12079, pp. 412-419). SPIE.

https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12079/120791Q/National-Basketball-Association-Most-Valuable-Player-prediction-based-on-machine/10.1117/12.2623094.full?SSO=1

[3] Nguyen, N. H., Nguyen, D. T. A., Ma, B., & Hu, J. (2021). The application of machine learning and deep learning in sport: predicting NBA players’ performance and popularity. Journal of Information and Telecommunication, 1-19.

https://www.tandfonline.com/doi/pdf/10.1080/24751839.2021.1977066?needAccess=true

### Gantt Chart Schedule
[Schedule and Roles](https://docs.google.com/spreadsheets/d/1LuZolhejX2NLJRxYQ-LqDw4Ic4dFY-k-Nkb-kCGmC5Q/edit?usp=sharing)

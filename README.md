# -Felicity_Machine_Learning_Competition

## Problem Statement
Dota2 is a free-to-play multiplayer online battle arena (MOBA) video game, which is played in matches between two teams of five players occupying and defending their own separate base on the map. Each player(user) can independently control one powerful character(hero) who has unique abilities and skills.  
**Given dataset of professional Dota players and their most frequent 10 heroes, we need to build a model to predict the performance(kda_ratio) of specific user-hero pair.**

## Dataset
Train and test datasets, from different set of users, contain user-hero pairs information. They are divided into two datasets respectively(train9.csv & train1.csv and test9.csv & test1.csv). 1 means one of the user's ten most frequent heros and the residuals are 9, which is chosen randomly. The aim is to predict the kda_ratio in test1.csv.
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/user_role_feature.PNG)  
We also have "hero_data.csv" which contains information about heros.
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/role_feature.PNG)

## Method
### Feature Engineering
We use train9.csv, train1.csv and test9.csv as training dataset to do feature engineering, which are shown below:
1. **Users**  
Assuming there is a between-users variance, we took Users’
self-performance in account. KDA ratio may be influenced by users’
effort, skills or mindset etc., so we calculate users’ total number of
games, total win games and mean win ratio to represent it.
2. **Heroes**
We think each hero has his/her pros and cons. However, some
heroes are probably meta for current version, so they are especially good
to use or easy to dominate the games with higher KDA ratio. We
calculate heroes’ total number of games, total win games and mean win
ratio to measure it.
Additionally, we grouped heroes into several types by their abilities
like base_health, base_str and base agi etc. Also, we calculate each hero
group’s total number of games, total win games and mean win ratio to
represent group performance.
3. **Interaction of Users and Heroes**  
Not only consider users’ and heroes ’features respectively, we also
take their interaction effect into account. Users play different types of
heroes may have different performance, especially when some kind of
heroes fit users’ potential well. As a result, given different primary_attr
and hero groups, we calculate their total number of games, total win
games and mean win ratio to measure interaction effect.
### Modeling
About the model selection, we choose **XGboost** as our primary
model, and we are going to use grid search to find tune the model
performance, leverage the **GridSearchCV** in sklearn on the following two
parameters, n_estimators and max_depth in XGBoost model, and get the
best model parameters, n_estimators=800, max_depth=9. Finally, we get
the final submissions.
## Result
We are Team **NTUBusinessSchool** and get the **5th** prize in private leaderboard.
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/rank.PNG)
## Improvement
During the feature engineering process, some of the features are the
arg. Of the groups data, which might data peeking problem (the feature
contain the information of the response). And this prevent the model
represent the ground true, and make our cross validation error
inconsistent with the score in leaderboard.
Therefore, we should introduce techniques like cross feature
engineering. Just like the original training data, we have train1 and train9,
we should use the train0 data to calculate the feature for train1, and
replace the data in train1 from train9 9 times to make all the training
data have their correct feature values.
## Reproduce
The steps you need to do for final submission are shown below:
1. set the root path in dota2_DataManipulation.py
2. put the training, testing and submission data in root path
3. run 'pip install -r requirements.txt'
4. run 'python dota2_DataManipulation.py'
5. get the submission file, submit.csv, on the root folder

## Reference
Felicity : Kings of Machine Learning: https://datahack.analyticsvidhya.com/contest/kings-of-machine-learning/

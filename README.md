# -Felicity_Machine_Learning_Competition
The competition was held by **Analytics Vidhya**, a data science competition platform in India.I teamed up with **Lawrencesiao**. He is my best teammate and mentor in data science. Without his modeling skill, we cannot reach the top rank. If you have any interest, you can see his github here(https://github.com/lawrencesiao)
## Problem Statement
Dota2 is a free-to-play multiplayer online battle arena (MOBA) video game, which is played in matches between two teams of five players occupying and defending their own separate base on the map. Each player(user) can independently control one powerful character(hero) who has unique abilities and skills.  
**Given dataset of professional Dota players and their most frequent 10 heroes, we need to build a model to predict the performance(kda_ratio) of specific user-hero pair.**

## Dataset
Train and test datasets, from different set of users, contain user-hero pairs information. They are divided into two datasets respectively(train9.csv & train1.csv and test9.csv & test1.csv). 1 means one of the user's ten most frequent heros and the residuals are 9, which is chosen randomly. The aim is to predict the kda_ratio in test1.csv.
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/user_role_feature.PNG)    

We also have "hero_data.csv" which contains information about heros.   

![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/role_feature.PNG)

## Measure
1. The predictions will be evaluated on **RMSE**.
2. The public private split is **40:60**.
## Method
### Feature Engineering
We used train9.csv, train1.csv and test9.csv as training dataset to do feature engineering, which are shown below:
1. **Users**  
Assuming there was a between-users variance, we took Users’
self-performance into account. KDA ratio might be influenced by users’
effort, skills or mindset etc., so we calculate users’ total number of
games, total win games and mean win ratio to represent it.
2. **Heroes**
We thought each hero has his/her pros and cons. However, some
heroes were probably meta for current version, so they were especially good
to use or easy to dominate the games with higher KDA ratio. We
calculated heroes’ total number of games, total win games and mean win
ratio to measure it.
Additionally, we grouped heroes into several types by their abilities
like base_health, base_str and base agi etc. Also, we calculated each hero
group’s total number of games, total win games and mean win ratio to
represent group performance.
3. **Interaction of Users and Heroes**  
Not only considered users’ and heroes ’features respectively, we also
took their interaction effect into account. Users with different types of
heroes have different performance, especially when some kind of
heroes fitted users’ potential well. As a result, given different primary_attr
and hero groups, we calculated their total number of games, total win
games and mean win ratio to measure interaction effect.
### Modeling
About the model selection, we chose **XGboost** as our primary
model, and we were going to use grid search to fine-tune the model
performance, leverage the **GridSearchCV** in sklearn on the following two
parameters, n_estimators and max_depth in XGBoost model, and get the
best model parameters, n_estimators=800, max_depth=9. Finally, we got
the final submissions.
## Result
We are Team **NTUBusinessSchool** and got the **5th** prize in private leaderboard.  

![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/public_rank.PNG)
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/rank.PNG)
## Improvement
During the feature engineering process, some of the features were the
arg. Of the groups data, which might **data peeking problem** (the feature
contain the information of the response). And this prevented the model
from representing the ground true, and made our cross validation error
inconsistent with the score in public leaderboard.
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

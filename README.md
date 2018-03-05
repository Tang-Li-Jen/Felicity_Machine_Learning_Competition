# -Felicity_Machine_Learning_Competition

## Problem Statement
Dota2 is a free-to-play multiplayer online battle arena (MOBA) video game, which is played in matches between two teams of five players occupying and defending their own separate base on the map. Each player(user) can independently control one powerful character(hero) who has unique abilities and skills.  
**Given dataset of professional Dota players and their most frequent 10 heroes, we need to build a model to predict the performance(kda_ratio) of specific user-hero pair.**

## Dataset
Train and test datasets, from different set of users, contain user-hero pairs information. They are divided into two datasets respectively(train9.csv & train1.csv and test9.csv & test1.csv). 1 means one of the user's ten most frequent heros and the residuals are 9, which is chosen randomly. The aim is to predict the kda_ratio in test1.csv.
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/user_role_feature.PNG)

We also have "hero_data.csv" which contains information about heros.
![image](https://github.com/Tang-Li-Jen/Felicity_Machine_Learning_Competition/blob/master/images/role_feature.PNG)

## Reproduce
The steps you need to do for final submission are shown below:
1. set the root path in dota2_DataManipulation.py
2. put the training, testing and submission data in root path
3. run 'pip install -r requirements.txt'
4. run 'python dota2_DataManipulation.py'
5. get the submission file, submit.csv, on the root folder

## Reference
Felicity : Kings of Machine Learning: https://datahack.analyticsvidhya.com/contest/kings-of-machine-learning/

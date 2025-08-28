 it is an unsupervised learning technique 

the goal is to discover relationships between variables in large datasets 

RULE LEARNING 
1. Support - frequency of an itemset in the dataset 
2. Confidence - Strength of implication 
3. Lift - importance of the rule 

APRIORI ALGORITHM 
most famous algorithm for association rule mining 
Steps of Apriori:
Set minimum support & confidence.
Generate itemsets of size 1, calculate their support.
Keep only itemsets ≥ minimum support.
Expand itemsets (size 2, size 3 …).
From frequent itemsets, generate rules and filter by confidence & lift.

Frequent Pattern Growth 
improvement over Apriori 
uses a tree structure 

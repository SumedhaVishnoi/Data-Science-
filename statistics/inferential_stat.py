import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# central limit theorem
pop_data = [np.random.randint(10,100) for i in range (10000)]
pop_table =  pd.DataFrame({"Pop_data": pop_data})
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from predictor import predictRuns


"""
sys.argv[1] is the input test file name given as command line arguments

"""
runs = predictRuns('inputFile.csv')
print("Predicted Runs: ", runs)


# In[ ]:





# In[ ]:




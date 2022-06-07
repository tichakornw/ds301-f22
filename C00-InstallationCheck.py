#!/usr/bin/env python
# coding: utf-8

# **DS 301: Applied Data Modeling and Predictive Analysis**
#
# **Homework 1 – Check installation of required packages**
#
# # Check Installation of Required Packages
# Nok Wongpiromsarn, 8 August 2022

# In[ ]:


# Python ≥3.7 is required
import sys
assert sys.version_info >= (3, 5)

# scikit-Learn ≥0.22 is required
import sklearn
assert sklearn.__version__ >= "0.22"

# pandas ≥1.0.5 is required
import pandas as pd
assert pd.__version__ >= "1.0.5"

# matplotlib ≥3.2 is required
import matplotlib
assert matplotlib.__version__ >= "3.2"

# seaborn ≥0.10 is required
import seaborn
assert seaborn.__version__ >= "0.10"

# tensorflow ≥2.3 is required
import tensorflow
assert tensorflow.__version__ >= "2.3"


# In[ ]:

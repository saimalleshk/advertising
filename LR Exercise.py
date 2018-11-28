
# coding: utf-8

# ## Logistic Regression Project using Advertising dataset
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on ad based off the features of that user.
# 
# The dataset contains the following features.
# 
# - 'Daily Time Spent on Site': Consumer time on site in minutes.
# - 'Age': Customer age in years.
# - 'Area Income': Avg. Income of geographical area of consumer
# - 'Daily Internet Usage': Avg.minutes a day consumer is on the internet
# - 'Ad Topic Line': Headline of the advertisement
# - 'City': City of consumer
# - 'Male': Whether or not consumer was male
# - 'Country': Country of consumer.
# - 'Timestamp': Time at which consumer clicked on Ad or closed window.
# - 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ### Importing Libraries
#  

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the data
# 
# **Read the advertising.csv file and set it to ad data frame.**

# In[3]:


ad_data = pd.read_csv('advertising.csv')


# ### Check the head of ad_data

# In[4]:


ad_data.head()


# ### Use info and describe on ad_data 

# In[5]:


ad_data.info()


# In[6]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# **Create a histogram of the Age**

# In[11]:


sns.set_style('whitegrid')
ad_data['Age'].plot.hist(bins=30)


# ### Create a jointplot showing Area income versus Age.

# In[12]:


sns.jointplot(x="Age", y="Area Income", data=ad_data)


# ### Create a jointplot showing the kde distributions of Daily Time Spent on site vs Age.

# In[13]:


sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data,kind='kde',color='red')


# ### Create a jointplot od 'Daily Time Spent on site' vs 'Daily Internet Usage'

# In[16]:


sns.jointplot(x="Daily Time Spent on Site",y="Daily Internet Usage",data=ad_data,color='green')


# ### Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.  

# In[17]:


sns.pairplot(ad_data, hue='Clicked on Ad')


# ## Logistic Regression
# 
# Now it's tiem to do a trian test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train out!
# 
# **Split the data into training set and testing set using train_test_split**

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


ad_data.head()


# In[22]:


ad_data.columns


# In[44]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Train and fit a logistic regression model on the training set.

# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


logmodel = LogisticRegression()


# In[47]:


logmodel.fit(X_train, y_train)


# ### Predictions and Evaluations
# 
# **Now Predict values for the testing data.**

# In[48]:


predictions = logmodel.predict(X_test)


# ### Create a classification report for the model .

# In[50]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))


# ## Conclusion
# 
# Since we have classified which age are spending more time and all. Now with the logistic model we can see that the model is working fine with good precision rate.

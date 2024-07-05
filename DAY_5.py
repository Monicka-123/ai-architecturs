#!/usr/bin/env python
# coding: utf-8

#                              LINEAR_REGRESSION

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


df = pd.read_excel(r'C:\Users\mmoni\OneDrive\Documents\DAY_5.xlsx')


# In[12]:


print(df)


# In[14]:


plt.scatter(df.area,df.price,color='red',marker='+')
plt.title("Home_PRICE")
plt.xlabel("area")
plt.ylabel("price")
plt.show()


# In[15]:


from sklearn.linear_model import LinearRegression
reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[16]:


reg.predict([[72000]])


# In[19]:


plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.title("Home_PRICE")
plt.xlabel("area")
plt.ylabel("price")
plt.show()


#                                       LOGISTIC_REGRESSION

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


db = pd.read_excel(r'C:\Users\mmoni\OneDrive\Documents\DAY_5_2.xlsx')
print(db)


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(db[['age']],db. yn ,test_size=0.2)


# In[31]:


x_train


# In[34]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[35]:


model.predict([[14]])


# In[38]:


plt.scatter(db.age,db.yn,color='red',marker='+')
plt.title("loan_pred")
plt.xlabel("age")
plt.ylabel("yn")
plt.show()


# DECISION TREE

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


dc = pd.read_excel(r'C:\Users\mmoni\OneDrive\Documents\DAY_5_3.xlsx')
print(dc)


# In[6]:


x= dc.drop('salary', axis='columns')

y=dc['salary']


# In[7]:


from sklearn.preprocessing import LabelEncoder
for_company= LabelEncoder()
for_job =LabelEncoder()
for_degree= LabelEncoder()



# In[9]:


x['company_n']= for_company.fit_transform(x['company']) 

x['job_n']= for_job.fit_transform(x['job'])

x['degree_n']= for_degree.fit_transform(x['degree'])

x.head()


# In[10]:


new=x.drop(['company','job','degree'],axis='columns')
print(new)


# In[14]:


from sklearn import tree


# In[15]:


model=tree.DecisionTreeClassifier()


# In[16]:


model.fit(new,y)


# In[17]:


model.score(new,y)
model.predict([[2,0,1]])


# In[ ]:





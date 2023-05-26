#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats
from scipy.stats import chi2_contingency 
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


df=pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


# Converting Created and Closed Date into datetime format
df["Created Date"]=pd.to_datetime(df["Created Date"])
df["Closed Date"]=pd.to_datetime(df["Closed Date"])


# In[6]:


#Creating the new column as "Req_Close_Time"
df["Req_Close_Time"]=(df["Closed Date"]-df["Created Date"])
Req_Close_Time=[]
for x in (df["Closed Date"]-df["Created Date"]):
    close=x.total_seconds()/60
    Req_Close_Time.append(close)
    
df["Req_Close_Time"]=Req_Close_Time


# In[7]:


df["Agency"].unique()


# In[8]:


sns.distplot(df["Req_Close_Time"])
plt.show


# In[13]:


df['Complaint Type'].value_counts()[:10].plot(kind='bar',color=list('rgbkymc'),alpha=0.7,figsize=(20,6))
plt.show()


# In[15]:


#To understand which type of complaints are taking more time to get resolved
sam=sns.catplot(x='Complaint Type', y="Req_Close_Time",data=df)
sam.fig.set_figwidth(15)
sam.fig.set_figheight(7)
plt.xticks(rotation=90)
plt.ylim((0,5000))
plt.show()


# In[20]:


#status of the requests
df['Status'].value_counts().plot(kind='bar',color=list('rgbkymc'),alpha=0.6,figsize=(20,6))
plt.show()


# In[21]:


#for Coloumn Borough
plt.figure(figsize=(12,7))
df['Borough'].value_counts().plot(kind='bar',alpha=0.7,color=list('rgbkymc'),figsize=(20,6))
plt.show()


# In[22]:


df["Location Type"].unique()


# In[23]:


pd.DataFrame(df.groupby("Location Type")["Req_Close_Time"].mean()).sort_values("Req_Close_Time")


# In[24]:


pd.DataFrame(df.groupby("City")["Req_Close_Time"].mean()).sort_values("Req_Close_Time")


# In[25]:


#percentage of missng values
pd.DataFrame((df.isnull().sum()/df.shape[0]*100)).sort_values(0,ascending=False)[:20]


# In[26]:


#Removing column with high percentage 
new_df=df.loc[:,(df.isnull().sum()/df.shape[0]*100)<=50]


# In[27]:


print("Old DataFrame Shape :",df.shape)
print("New DataFrame Shape : ",new_df.shape)


# In[28]:


usp=[]
for x in new_df.columns.tolist():
    if new_df[x].nunique()<=3:
        print(x+ " "*10+" : ",new_df[x].unique())
        usp.append(x)


# In[29]:


new_df.drop(usp,axis=1,inplace=True)


# In[30]:


new_df.shape


# In[31]:


usp1=["Unique Key","Incident Address","Descriptor","Street Name",
      "Cross Street 1","Cross Street 2","Due Date","Resolution Description",
      "Resolution Action Updated Date","Community Board","X Coordinate (State Plane)",
      "Y Coordinate (State Plane)","Park Borough","Latitude","Longitude","Location"]


# In[32]:


#Removing unnecessasory columns
new_df.drop(usp1,axis=1,inplace=True)


# In[34]:


new_df.head(10)


# In[35]:


sam=sns.catplot(x="Complaint Type",y="Req_Close_Time",kind="box",data=new_df)
sam.fig.set_figheight(8)
sam.fig.set_figwidth(15)
plt.xticks(rotation=90)
plt.ylim((0,2000))


# In[36]:


df1=pd.DataFrame()
df1["Req_Close_Time"]=new_df["Req_Close_Time"]
df1["Complaint"]=new_df["Complaint Type"]


# In[37]:


df1.dropna(inplace=True)


# In[38]:


df1.head()


# In[39]:


lm=ols("Req_Close_Time~Complaint",data=df1).fit()
table=sm.stats.anova_lm(lm)
table


# In[40]:


chi_sq=pd.DataFrame()
chi_sq["Location Type"]=new_df["Location Type"]
chi_sq["Complaint Type"]=new_df["Complaint Type"]


# In[41]:


chi_sq.dropna(inplace=True)


# In[42]:


ctab = pd.crosstab( chi_sq["Location Type"],chi_sq["Complaint Type"])


# In[43]:


stat, p, dof, expected = chi2_contingency(ctab) 
alpha = 0.05
if p <= alpha: 
    print('Dependent (Reject H0)') 
else: 
    print('Independent (H0 Holds True)')


# In[ ]:





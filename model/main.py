#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


df = pd.read_csv('mldata.csv')
df.head()


# In[3]:


print('The shape of our training set: %s professionals and %s features'%(df.shape[0],df.shape[1]))


# Data Preprocessing

# In[4]:


print("Columns in our dataset: " , df.columns)


# In[5]:


print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())
print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())


# Checking Missing Values

# In[6]:


df.isnull().sum(axis=0)


# **Observation: No missing values.**

# Distinct Values for Categorical Features

# In[7]:


categorical_col = df[['self-learning capability?', 'Extra-courses did','reading and writing skills', 'memory capability score', 
                      'Taken inputs from seniors or elders', 'Management or Technical', 'hard/smart worker', 'worked in teams ever?', 
                      'Introvert', 'interested career area ']]
for i in categorical_col:
    print(df[i].value_counts(), end="\n\n")


# Data Balancing for Classification

# In[8]:


sns.set(rc={'figure.figsize':(50,10)})
sns.countplot(x = df["Suggested Job Role"])


# Correlation Between Numerical Features

# In[9]:


corr = df[['Logical quotient rating', 'hackathons', 
           'coding skills rating', 'public speaking points']].corr()
f,axes = plt.subplots(1,1,figsize = (10,10))
sns.heatmap(corr,square=True,annot = True,linewidth = .4,center = 2,ax = axes)


# No highly corelated numerical pair found

# Visualization for Categorical Variables

# In[10]:


print(df["Interested subjects"].value_counts())


# In[11]:


# Figure Size
fig, ax = plt.subplots(figsize=(12,6))

# Horizontal Bar Plot
title_cnt=df["Interested subjects"].value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1],edgecolor='black', color=sns.color_palette('pastel',len(title_cnt)))




# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Interested Subjects',weight='bold',fontsize=20)
ax.set_xlabel('Count', weight='bold')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# Show Plot
plt.show()


# In[12]:


print(df["certifications"].value_counts())


# In[13]:


# Figure Size
fig, ax = plt.subplots(figsize=(12,6))

# Horizontal Bar Plot
title_cnt=df.certifications.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1],edgecolor='black', color=sns.color_palette('pastel',len(title_cnt)))



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Certifications',weight='bold',fontsize=20)
ax.set_xlabel('Count', weight='bold')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# Show Plot
plt.show()


# In[14]:


print(df["Type of company want to settle in?"].value_counts())


# In[15]:


# Figure Size
fig, ax = plt.subplots(figsize=(12,6))

# Horizontal Bar Plot
title_cnt=df["Type of company want to settle in?"].value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1],edgecolor='black', color=sns.color_palette('pastel',len(title_cnt)))



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Type of Company you want to settle in?',weight='bold',fontsize=20)
ax.set_xlabel('Count', weight='bold')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# Show Plot
plt.show()


# In[16]:


print(df["interested career area "].value_counts())


# In[17]:


# Figure Size
fig, ax = plt.subplots(figsize=(10,4)) #width,height

# Horizontal Bar Plot
title_cnt=df["interested career area "].value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1],edgecolor='black', color=sns.color_palette('pastel',len(title_cnt)))



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)


# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Interested Career Area ',weight='bold',fontsize=20)
ax.set_xlabel('Count', weight='bold')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')

# Show Plot
plt.show()


# Binary Encoding for Categorical Variables

# In[18]:


cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert","Management or Technical","hard/smart worker"]]
for i in cols:
    cleanup_nums = {i: {"yes": 1, "no": 0, "smart worker": 1, "hard worker": 0, "Management": 1, "Technical": 0}}

    df = df.replace(cleanup_nums)


# Number Encoding for Categorical 

# In[19]:


mycol = df[['reading and writing skills', 'memory capability score', 'certifications', 'Management or Technical',
            'hard/smart worker', 'Type of company want to settle in?', 'Interested subjects', 'interested career area ']]
for i in mycol:
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2, "r programming": 1, "information security": 2,
                        "shell programming": 3, "machine learning": 4, "full stack": 5, "hadoop": 6, "python": 7, 
                        "distro making": 8, "app development": 9, "Service Based": 1, "Web Services": 2,
                        "BPA": 3, "Testing and Maintainance Services": 4, "Product based": 5, "Finance": 6, "Cloud Services": 7, 
                        "product development": 8, "Sales and Marketing": 9, "SAaS services": 10, "system developer": 1, "security": 2,
                        "Business process analyst": 3, "developer": 4, "testing": 5, "cloud computing": 6,
                        "Software Engineering": 1, "IOT": 2, "cloud computing domain": 3, "programming": 4, "networks": 5, "Computer Architecture": 6, "data engineering": 7, 
                        "hacking": 8, "Management": 9, "parallel computing": 10}}
    df = df.replace(cleanup_nums)


print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())


# Dummy Variable Encoding

# In[20]:


df.head()


# In[21]:


print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


# Building Machine Learning Model

# In[22]:


feed = df[['Logical quotient rating', 'hackathons', 'coding skills rating',
       'public speaking points', 'self-learning capability?',
       'Extra-courses did', 'certifications', 'reading and writing skills',
       'memory capability score', 'Interested subjects',
       'interested career area ', 'Type of company want to settle in?',
       'Taken inputs from seniors or elders', 'Management or Technical',
       'hard/smart worker', 'worked in teams ever?', 'Introvert',
             'Suggested Job Role']]

# Taking all independent variable columns
df_train_x = feed.drop('Suggested Job Role',axis = 1)

# Target variable column
df_train_y = feed['Suggested Job Role']

x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.40, random_state=62)


# Decision Tree Classifier

# In[23]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)




y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("accuracy=",accuracy*100)


# Predicting class

# In[24]:


userdata = [['5','0','6','2','1','0','2','0','0','4','5','3','0','1','1','1','0']]
ynewclass = clf.predict(userdata)
ynew = clf.predict_proba(userdata)
print(ynewclass)
print("Probabilities of all classes: ", ynew)
print("Probability of Predicted class : ", np.max(ynew))


# In[25]:


pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


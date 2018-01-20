
# coding: utf-8

# # Gramener-Product Team Hiring Exercise
# 

# # Python:
# ### Given two lists L1 = ['a', 'b', 'c'], L2 = ['b', 'd'], find common elements, find elements present in L1 and not in L2?
# 
# ### How many Thursdays were there between 1990 - 2000?
# 
# ### Given the following Javascript array:
# 
# ### var data = [0, 1, 2, 'stop', 2, 0, 1, 'stop']
# 
# ### write a Javascript function or expression that returns an array with just the zeroes removed.
# 

# In[154]:

# 1. Given two lists L1 = ['a', 'b', 'c'], L2 = ['b', 'd'], find common elements, find elements present in L1 and not in L2?

#  Solution
   #Printing Common elements 
L1 = ['a', 'b', 'c']
L2 = ['b', 'd']
print list(set(L1).intersection(set(L2)))
               # Or
print list(set(L1) & (set(L2)))

   #Printing elements in L1 not in L2
print [x for x in L1 if x not in L2]
             #or
print filter(lambda x: x not in L2, L1)

# 2. How many Thursdays were there between 1990 - 2000?
from datetime import datetime
from dateutil import rrule

print len(list(rrule.rrule(rrule.DAILY,
                         dtstart=datetime(1990, 1, 1),
                         until=datetime(2000, 12, 31),
                         byweekday=[rrule.TH])))



from datetime import date
import datetime
import calendar
d0 = date(1990, 1, 1)
d1 = date(2000, 12, 31)
delta = d0 - d1
print delta.days
print list(calendar.day_abbr)
print calendar.day_abbr[3] 

from datetime import datetime
sum(datetime(year, month, 1).weekday() == 4
      for year in range(1950, 2051) for month in range(1,13))


# # Use case 1 - National Achievement Survey 
# 
# ### By Gopala Krishna Chaitanya Y
# 

# # 1. What influences students performance the most?

# In[130]:

#Loding and importing all the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.plotly as py
import plotly.graph_objs as go
from scipy import stats
from sklearn import preprocessing as prep
from sklearn.preprocessing import Imputer
import sklearn as sk;
import pylab as pl
plt.style.use('ggplot')
get_ipython().magic(u'matplotlib notebook')
get_ipython().magic(u'matplotlib inline')
# suppress all warnings
import warnings
warnings.filterwarnings('ignore')


# In[131]:

#Reading the data from CSV files and imputing mean value for all the subject marks
pupil_marks = pd.read_csv('gramener-usecase-nas/nas-pupil-marks.csv')
labels = pd.read_csv('gramener-usecase-nas/nas-labels.csv')
pupil_marks.isnull().sum()
pupil_marks['Maths %'].fillna((pupil_marks['Maths %'].mean()), inplace=True)
pupil_marks['Reading %'].fillna((pupil_marks['Reading %'].mean()), inplace=True)
pupil_marks['Science %'].fillna((pupil_marks['Science %'].mean()), inplace=True)
pupil_marks['Social %'].fillna((pupil_marks['Social %'].mean()), inplace=True)


# In[132]:

#Mapping the Categorical variables to numerical as we can perform analysis on these features once transformed
pupil_marks['Use computer'] = pupil_marks['Use computer'].map({"Yes":1,"No":0})
pupil_marks['Subjects'] = pupil_marks['Subjects'].map({'L':1, 'S':2, 'O':3, 'M':4, '0':0})
pupil_marks['Use computer']
for col in ['Use computer', 'Subjects']:
    pupil_marks[col] = pupil_marks[col].astype('category')
pupil_marks = pupil_marks.fillna(pupil_marks.median())
#checking if there are any NaNs in the data
pupil_marks.isnull().sum()


# In[133]:

#Calcualting average of all the subject marks to create and calculate the feature 'Performance' 
summary_ave_data = pupil_marks[['Maths %', 'Reading %', 'Science %', 'Social %']]
pupil_marks['Performance'] = summary_ave_data.mean(axis=1)
pupil_marks['Performance'].describe()
pupil_marks.head(10)


# In[134]:

# Here I have created new dataframe with all the features to analyse
Pupil_features=['Gender', 'Age', 'Category',
       'Same language', 'Siblings', 'Handicap', 'Father edu', 'Mother edu',
       'Father occupation', 'Mother occupation', 'Below poverty',
       'Use calculator', 'Use computer', 'Use Internet', 'Use dictionary',
       'Read other books', '# Books', 'Distance', 'Computer use',
       'Library use', 'Like school', 'Subjects', 'Give Lang HW',
       'Give Math HW', 'Give Scie HW', 'Give SoSc HW', 'Correct Lang HW',
       'Correct Math HW', 'Correct Scie HW', 'Correct SocS HW',
       'Help in Study', 'Private tuition', 'English is difficult',
       'Read English', 'Dictionary to learn', 'Answer English WB',
       'Answer English aloud', 'Maths is difficult', 'Solve Maths',
       'Solve Maths in groups', 'Draw geometry', 'Explain answers',
       'SocSci is difficult', 'Historical excursions', 'Participate in SocSci',
       'Small groups in SocSci', 'Express SocSci views',
       'Science is difficult', 'Observe experiments', 'Conduct experiments',
       'Solve science problems', 'Express science views', 'Watch TV',
       'Read magazine', 'Read a book', 'Play games', 'Help in household']

Pupil_feature_data = pd.DataFrame(pupil_marks,columns=Pupil_features)


# In[136]:

#Running linear Regression on Performance to see how is it dependent
import statsmodels.api as sm
from sklearn import linear_model
X = Pupil_feature_data
y = pupil_marks['Performance']
lm = linear_model.LinearRegression()
linmodel = lm.fit(X,y)
predictions = linmodel.predict(X)
print(predictions)
print(linmodel.score(X,y))
linmodel


# In[12]:

#Running ElasticNet model to find the dependency of Performance
import statsmodels.api as sm
from sklearn import linear_model
X = Pupil_feature_data
y = pupil_marks['Performance']
model_ElasticNet = linear_model.ElasticNet()
model_ElasticNet.fit(X, y)
print (model_ElasticNet.score(X, y))


# In[137]:

# Splitting the data to training and test data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

train, test = train_test_split(Pupil_feature_data, test_size=0.2)


# In[14]:

# Created  Random Forest Regressor model that identifies the most influencing feature on the Overal Performance of the student
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
 
X = Pupil_feature_data.values
Y = pupil_marks['Performance'].values
names = Pupil_features
 
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print sorted(scores, reverse=True)
scoresdf=pd.DataFrame(sorted(scores, reverse=True)) 
scoresdf.columns= ["Scores of Importance", "Features"]
scoresdf


# In[95]:




# In[16]:


import matplotlib.pyplot as plt
my_colors = [(x/0.010, x/0.020, 0.75) for x in range(len(scoresdf))]
ax = scoresdf[['Features','Scores of Importance']].plot(kind='bar', title ="Importance of Features", use_index=False,color=my_colors, figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Scores of Importance", fontsize=12)
ax.set_xticklabels(scoresdf['Features'])
plt.show()


# In[105]:

# Created  Random Forest Regressor model that identifies the most influencing feature on the Performance in Maths of the student
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
 
S = Pupil_feature_data.values
T = pupil_marks['Maths %'].values
names = Pupil_features
 
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
Mathsscores = []
for i in range(S.shape[1]):
     Mathsscore = cross_val_score(rf, S[:, i:i+1], T, scoring="r2",
                              cv=ShuffleSplit(len(S), 3, .3))
     Mathsscores.append((round(np.mean(Mathsscore), 3), names[i]))
print sorted(Mathsscores, reverse=True)
Mathsscoresdf=pd.DataFrame(sorted(Mathsscores, reverse=True)) 
Mathsscoresdf.columns= ["Maths Scores of Importance", "Features"]
Mathsscoresdf


# In[101]:

#Plotting a bar graph comparing the features Highest influence to lowest
import matplotlib.pyplot as plt
my_colors = [(x/0.010, x/0.020, 0.75) for x in range(len(Mathsscoresdf))]
ax = Mathsscoresdf[['Features','Maths Scores of Importance']].plot(kind='bar', title ="Importance of Features", use_index=False,color=my_colors, figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Maths Scores of Importance", fontsize=12)
ax.set_xticklabels(Mathsscoresdf['Features'])
plt.show()


# In[100]:

# Created  Random Forest Regressor model that identifies the most influencing feature on the Performance in Reading of the student
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
 
A = Pupil_feature_data.values
B = pupil_marks['Reading %'].values
names = Pupil_features
 
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
Readingscores = []
for i in range(A.shape[1]):
     Readingscore = cross_val_score(rf, A[:, i:i+1], B, scoring="r2",
                              cv=ShuffleSplit(len(A), 3, .3))
     Readingscores.append((round(np.mean(Readingscore), 3), names[i]))
print sorted(Readingscores, reverse=True)
Readingscoresdf=pd.DataFrame(sorted(Readingscores, reverse=True)) 
Readingscoresdf.columns= ["Reading Scores of Importance", "Features"]
Readingscoresdf


# In[102]:

#Plotting a bar graph comparing the features Highest influence to lowest
import matplotlib.pyplot as plt
my_colors = [(x/0.010, x/0.020, 0.75) for x in range(len(Readingscoresdf))]
ax = Readingscoresdf[['Features','Reading Scores of Importance']].plot(kind='bar', title ="Importance of Features", use_index=False,color=my_colors, figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Reading Scores of Importance", fontsize=12)
ax.set_xticklabels(Readingscoresdf['Features'])
plt.show()


# In[113]:

# Created  Random Forest Regressor model that identifies the most influencing feature on the Performance in Science of the student
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
 
P = Pupil_feature_data.values
Q = pupil_marks['Science %'].values
names = Pupil_features
 
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
Sciencescores = []
for i in range(P.shape[1]):
     Sciencescore = cross_val_score(rf, P[:, i:i+1], Q, scoring="r2",
                              cv=ShuffleSplit(len(P), 3, .3))
     Sciencescores.append((round(np.mean(Sciencescore), 3), names[i]))
print sorted(Sciencescores, reverse=True)
Sciencescoresdf=pd.DataFrame(sorted(Sciencescores, reverse=True)) 
Sciencescoresdf.columns= ["Science Scores of Importance", "Features"]
Sciencescoresdf


# In[114]:

#Plotting a bar graph comparing the features Highest influence to lowest
import matplotlib.pyplot as plt
my_colors = [(x/0.010, x/0.020, 0.75) for x in range(len(Sciencescoresdf))]
ax = Sciencescoresdf[['Features','Science Scores of Importance']].plot(kind='bar', title ="Importance of Features", use_index=False,color=my_colors, figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Science Scores of Importance", fontsize=12)
ax.set_xticklabels(Sciencescoresdf['Features'])
plt.show()


# In[115]:

# Created  Random Forest Regressor model that identifies the most influencing feature on the Performance in Social of the student
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
 
C = Pupil_feature_data.values
D = pupil_marks['Social %'].values
names = Pupil_features
 
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
Socialscores = []
for i in range(P.shape[1]):
     Socialscore = cross_val_score(rf, C[:, i:i+1], D, scoring="r2",
                              cv=ShuffleSplit(len(C), 3, .3))
     Socialscores.append((round(np.mean(Socialscore), 3), names[i]))
print sorted(Socialscores, reverse=True)
Socialscoresdf=pd.DataFrame(sorted(Socialscores, reverse=True)) 
Socialscoresdf.columns= ["Social Scores of Importance", "Features"]
Socialscoresdf


# In[118]:

#Plotting a bar graph comparing the features Highest influence to lowest
import matplotlib.pyplot as plt
my_colors = [(x/0.010, x/0.020, 0.75) for x in range(len(Socialscoresdf))]
ax = Socialscoresdf[['Features','Social Scores of Importance']].plot(kind='bar', title ="Importance of Features", use_index=False,color=my_colors, figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Social Scores of Importance", fontsize=12)
ax.set_xticklabels(Socialscoresdf['Features'])
plt.show()


# In[138]:

#creating a dataframe withscores and features that influence the Total Performance the most
df=scoresdf[['Scores of Importance','Features']]


# ### What influences students performance the most?
# 
# I started the analysis with basic data exploration and cleaning the data. The data of Marks has lot of NaN or Null values.So I have Imputed these values with mean of marks of all the students with non null values for each subject.  I have created a feature 'Performance' by calculating the average of all subjects ('Maths %', 'Reading %', 'Science %', 'Social %'). 
# 
# Now I have chosen Random Forest Regressor algorithm for most influencing feature selection which works by creating scores for each feature on how much is that feature influencing in the data. From the results of Random Forest Regressor algorithm I have found the best features influencing the students performance the most. Here is the list below 
# 
# |Parameter|Most Influencing Feature |
# | --- |:--- |
# |Total Performance|'Father edu'|
# |Performance in Maths|'Computer Use'|
# |Performance in Reading|'Mother edu'|
# |Performance in Science|'Father edu'|
# |Performance in Social|'Help in household'|
# 
# From the Graphs above we can see that individual Subjects marks are also influenced by many features mostly.
# some are  
# More Siblings in family influences more performance in both Reading % and Maths %
# Using Dictionary influences Reading %
# Language Hw influences Math %
# State influences Maths %
# 
# Also Every Subjects Performance is influenced by few common features as listed below.
# Father edu
# Mother edu
# help in Household
# Father occupation
# 
# Checking all the features, ‘Father edu’ stands out first with much higher score than the second most influency feature.

# In[111]:




# # 2. How do boys and girls perform across states?

# In[139]:

#Creating list of all States available in dataset
Statelist=pupil_marks['State'].unique()
Statelist


# In[140]:

#Creating a total performance of Boys and Girls Seperately grouped by each State
i=0
columns = ['State','Performance_of_Boys', 'Performance_of_Girls']
rows = []
while i < len(Statelist):
    BoyPerformancedataforstate= pupil_marks[(pupil_marks.State==Statelist[i]) & (pupil_marks.Gender==1)].Performance
    GirlPerformancedataforstate= pupil_marks[(pupil_marks.State==Statelist[i]) & (pupil_marks.Gender==2)].Performance
    TotalPerfofGirl = np.sum(GirlPerformancedataforstate)
    TotalPerfofBoy = np.sum(BoyPerformancedataforstate)
    row = [Statelist[i], TotalPerfofBoy, TotalPerfofGirl]
    rows.append(row)
    i = i + 1
df = pd.DataFrame(rows, columns=columns)
statesdf=labels[labels['Column']=="State"]
print(statesdf)
df


# In[22]:

#Created a bar plot for above calulated Performance of Boys and Girls Comparing them for each State
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
sorteddf=df.assign(f = df['Performance_of_Boys']+ df['Performance_of_Girls']).sort_values('f',ascending=False).drop('f', axis=1)
sdf_joinfull=sorteddf.set_index('State').join(statesdf.set_index('Name'))
ax = sorteddf[['Performance_of_Boys','Performance_of_Girls']].plot(kind='bar', title ="Performance of Boys Vs Girls across States", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("States", fontsize=12)
ax.set_ylabel("Total Performance", fontsize=12)
ax.set_xticklabels(sdf_joinfull['Rename'])
plt.show()


# In[141]:

#Here Filtered the obtained data on States where Boys Perform better than Girls. So in the Below plot we can see the data only for states where Boys perform better than Girls.
BoysGGirls=df[df.Performance_of_Boys>df.Performance_of_Girls]
sortedBoysGGirlsdf=BoysGGirls.assign(f = df['Performance_of_Boys']+ df['Performance_of_Girls']).sort_values('f',ascending=False).drop('f', axis=1)
sdf_joinBgG=sortedBoysGGirlsdf.set_index('State').join(statesdf.set_index('Name'))
import matplotlib.pyplot as plt
ax = sortedBoysGGirlsdf[['Performance_of_Boys','Performance_of_Girls']].plot(kind='bar', title ="Performance of Boys Vs Girls across States", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("States", fontsize=12)
ax.set_ylabel("Total Performance", fontsize=12)
ax.set_xticklabels(sdf_joinBgG['Rename'])
plt.show()


# In[142]:

#Here Filtered the obtained data on States where Girls Perform better than Boys. So in the Below plot we can see the data only for states where Girls perform better than Boys.
import matplotlib.pyplot as plt
GirlsGBoys = df[df.Performance_of_Girls>df.Performance_of_Boys]
sortedGirlsGBoysdf=GirlsGBoys.assign(f = df['Performance_of_Boys']+ df['Performance_of_Girls']).sort_values('f',ascending=False).drop('f', axis=1)
sdf_joinGgB=sortedGirlsGBoysdf.set_index('State').join(statesdf.set_index('Name'))
ax = sdf_joinGgB[['Performance_of_Boys','Performance_of_Girls']].plot(kind='bar', title ="Performance of Boys Vs Girls across States", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("States", fontsize=12)
ax.set_ylabel("Total Performance", fontsize=12)
ax.set_xticklabels(sdf_joinGgB['Rename'])
plt.show()


# # 3. Do students from South Indian states really excel at Math and Science?

# In[ ]:

#Here I  have normalized the data of Performance of Boys and Girls (Min-Max Normalization)
boys_max_value = df['Performance_of_Boys'].max()
boys_min_value = df['Performance_of_Boys'].min()
df['Normalized_Perf_Boys'] = (df['Performance_of_Boys'] - boys_min_value) / (boys_max_value - boys_min_value)
girls_max_value = df['Performance_of_Girls'].max()
girls_min_value = df['Performance_of_Girls'].min()
df['Normalized_Perf_Girls'] = (df['Performance_of_Girls'] - girls_min_value) / (girls_max_value - girls_min_value)
# I have generated and plotted linear model comparing Performance of Boys With Girls
import seaborn as sns
withStatenames=df.set_index('State').join(statesdf.set_index('Name')) 
sns.lmplot( x="Normalized_Perf_Boys", y="Normalized_Perf_Girls",  size=15, data=withStatenames, fit_reg=True, hue='Rename', legend=False)
plt.legend(loc='lower right')
sns.plt.show()


# In[153]:

# Here I have Seperated the States to North India and South India
NorthList = []
for k in range(pupil_marks.shape[0]):
        if pupil_marks.iloc[k]["State"] not in ['AP','GA','KA','KL','PY','TN']:
            NorthList.append(pupil_marks.iloc[k]["State"])
        else:
            NorthList.append("SOUTH")
pupil_marks["Sep_South_North"] = NorthList
pupil_marks


# In[145]:

#Here I have seperated the entire data available into South India and other states of India
pupil_marks["South"]=pupil_marks["State"].isin(['AP','GA','KA','KL','PY','TN'])
pupil_marks["South"] = pupil_marks["South"].replace({True: 'SOUTH INDIA', False: 'REST OF INDIA'})
pupil_marks['Average_of_Maths_Science'] = pupil_marks[['Maths %','Science %']].apply(np.nanmean,axis=1)
Summarydf=pupil_marks[['State','Maths %','Science %','Average_of_Maths_Science','South','Sep_South_North']].describe()
SouthGroupbydf=pupil_marks[['State','Maths %','Science %','Average_of_Maths_Science','South','Sep_South_North']].groupby(by = "South").describe()
NorthGroupbydf = pupil_marks[['State','Maths %','Science %','Average_of_Maths_Science','South','Sep_South_North']].groupby(by = "Sep_South_North").describe()
print(Summarydf)
print(NorthGroupbydf)


# In[146]:

#Here I have Grouped the avearge performance of Maths and Science of all students by South India and Rest Of the India for Comparison
import matplotlib.colors as colors
from matplotlib.cm import bwr as cmap
import matplotlib.patches as mpatches
plt.figure(figsize=(12,5))
pupil_marks['Average_of_Maths_Science'] = pupil_marks[['Maths %','Science %']].apply(np.nanmean,axis=1)
Maths_Sci_perfomance = pupil_marks.groupby(["Sep_South_North","South"]).mean()["Average_of_Maths_Science"].reset_index()
Maths_perfomance = pupil_marks.groupby(["Sep_South_North","South"]).mean()["Maths %"].reset_index()
Sci_perfomance = pupil_marks.groupby(["Sep_South_North","South"]).mean()["Science %"].reset_index()
fulljoined=pd.merge(pd.merge(Sci_perfomance,Maths_perfomance,on='Sep_South_North'),Maths_Sci_perfomance,on='Sep_South_North')
withStatenamesMat= pd.merge(fulljoined, statesdf, how='left',left_on=['Sep_South_North'], right_on=['Name'])
withStatenamesMat["Rename"]=withStatenamesMatSci["Rename"].fillna('SOUTH')

#Created Bar plot for comparison of Maths and Science Performances in South India and other states of India
import matplotlib.pyplot as plt
ax = withStatenamesMat[['Average_of_Maths_Science','Maths %','Science %']].plot(kind='bar', title ="Performance of Maths and Science \n across South India and Rest of India", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("States", fontsize=12)
ax.set_ylabel("Total Maths and Science", fontsize=12)
ax.set_xticklabels(withStatenamesMat['Rename'])
plt.show()


# In[78]:

#Here I have calculated and plotted the performance of Maths and Science for South India and Rest of India combining States
import matplotlib.colors as colors
from matplotlib.cm import bwr as cmap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
pupil_marks['Average_of_Maths_Science'] = pupil_marks[['Maths %','Science %']].apply(np.nanmean,axis=1)
Maths_Sci_perfomance = pupil_marks.groupby(["South"]).mean()["Average_of_Maths_Science"].reset_index()
Maths_perfomance = pupil_marks.groupby(["South"]).mean()["Maths %"].reset_index()
Sci_perfomance = pupil_marks.groupby(["South"]).mean()["Science %"].reset_index()
fulljoined=pd.merge(pd.merge(Sci_perfomance,Maths_perfomance,on='South'),Maths_Sci_perfomance,on='South'
ax = fulljoined[['Average_of_Maths_Science','Maths %','Science %']].plot(kind='bar', title ="Maths and Science Performance of North States Vs South States", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("States", fontsize=12)
ax.set_ylabel("Total Maths and Science", fontsize=12)
ax.set_xticklabels(fulljoined['South'])
plt.show()


# In[147]:

#Here I have normalized the data for Maths and Science performances and Created a linear model for comparison between South India and Rest of India in Maths and Science
Maths_max_value = fulljoined['Maths %'].max()
Maths_min_value = fulljoined['Maths %'].min()
fulljoined['Normalized_Perf_Maths'] = (fulljoined['Maths %'] - Maths_min_value) / (Maths_max_value - Maths_min_value)
science_max_value = fulljoined['Science %'].max()
science_min_value = fulljoined['Science %'].min()
fulljoined['Normalized_Perf_Science'] = (fulljoined['Science %'] - science_min_value) / (science_max_value - science_min_value)
import seaborn as sns
withStatenamesMatSci= pd.merge(fulljoined, statesdf, how='left',left_on=['Sep_South_North'], right_on=['Name'])
withStatenamesMatSci["Rename"]=withStatenamesMatSci["Rename"].fillna('SOUTH')
sns.lmplot( x="Normalized_Perf_Maths", y="Normalized_Perf_Science",  size=15, data=withStatenamesMatSci, fit_reg=True, hue='South',palette="Set1", legend=False)
plt.legend(loc='lower right')
sns.plt.show()


# In[129]:

#Scatter plot to show the performances of Maths and Science.
colors = np.where(withStatenamesMatSci['Rename'], 'r', 'k')
withStatenamesMatSci.plot(kind='scatter', x='Normalized_Perf_Maths',y='Normalized_Perf_Science', s=50, c='k')


# In[87]:

withStatenamesMatSci


# In[148]:

pupil_marks['Average_of_Maths_Science'] = pupil_marks[['Maths %','Science %']].apply(np.nanmean,axis=1)
Southperfomance = pupil_marks.groupby(["South"]).mean()["Average_of_Maths_Science"].reset_index()
Southperfomance


# In[152]:

#Plotted a bar graph comparing the Overall Performance of Students in Maths and Science between South India and Rest of India
ax = Southperfomance[['Average_of_Maths_Science','South']].plot(kind='bar', title ="Maths and Science Performance of Rest of India Vs South India", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("States", fontsize=12)
ax.set_ylabel("Total Maths and Science", fontsize=12)
ax.set_xticklabels(Southperfomance['South'])
plt.show()


# # 2. How do boys and girls perform across states?
# ## I have Created Features of Performance grouped by States by Gender (Boy, Girl) . Then I have calculated overall Performance ans average of all subjects Then I have Plotted the overall Performances Statewise comparing Performance of Boy Vs Girl. From The analysis I found that Girls Perform better than Boys
# 
# # 3.Do students from South Indian states really excel at Math and Science?
# ## I have created a feature of Avearge Performance in Maths and Science. I have segregated the states as South Indian States and Rest of India. Then I have Plotted the overall Performance in Maths and Science Comparing between south Indian Students and Student from Rest of India. From my Analysis I found that overall, North Indian Students are good at Maths and Science than South Indian Students

# In[ ]:




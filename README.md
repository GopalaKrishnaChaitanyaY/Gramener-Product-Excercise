# Gramener-Product-Excercise

1. What influences students performance the most?
I started the analysis with basic data exploration and cleaning the data. The data of Marks has lot of NaN or Null values.So I have Imputed these values with mean of marks of all the students with non null values for each subject.  I have created a feature 'Performance' by calculating the average of all subjects ('Maths %', 'Reading %', 'Science %', 'Social %'). 

Now I have chosen Random Forest Regressor algorithm for most influencing feature selection which works by creating scores for each feature on how much is that feature influencing in the data. From the results of Random Forest Regressor algorithm I have found the best features influencing the students performance the most. Here is the list below 

|Parameter|Most Influencing Feature |
| --- |:--- |
|Total Performance|'Father edu'|
|Performance in Maths|'Computer Use'|
|Performance in Reading|'Mother edu'|
|Performance in Science|'Father edu'|
|Performance in Social|'Help in household'|

From the Graphs above we can see that individual Subjects marks are also influenced by many features mostly.
some are  
More Siblings in family influences more performance in both Reading % and Maths %
Using Dictionary influences Reading %
Language Hw influences Math %
State influences Maths %

Also Every Subjects Performance is influenced by few common features as listed below.
Father edu
Mother edu
help in Household
Father occupation

Checking all the features, ‘Father edu’ stands out first with much higher score than the second most influency feature.

2. How do boys and girls perform across states?
I have Created Features of Performance grouped by States by Gender (Boy, Girl) . Then I have calculated overall Performance ans average of all subjects
Then I have Plotted the overall Performances Statewise comparing Performance of Boy Vs Girl.

3. Do students from South Indian states really excel at Math and Science?
I have created a feature of Avearge Performance in Maths and Science. I have segregated the states as South Indian States and Rest of India. Then I have Plotted the overall Performance in Maths and Science 
Comparing between south Indian Students and Student from Rest of India.

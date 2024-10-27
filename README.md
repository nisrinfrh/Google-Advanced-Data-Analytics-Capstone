## Introduction: 

Waze’s free navigation app makes it easier for drivers around the world to get to where they want to go. Waze’s community of map editors, beta testers, translators, partners, and users helps make each drive better and safer. 

## Project goal:  
I will develop a machine learning model to predict user churn. Churn quantifies the number of users who have uninstalled the Waze app or stopped using the app. This project focuses on monthly user churn. An accurate model will help prevent churn, improve user retention, and grow Waze’s business

The following tasks are needed before I begin the data analysis process:

Build a dataframe for the churn dataset

Examine data type of each column

Gather descriptive statistics

Determine the correct modeling approach

Build a regression model

Finish checking model assumptions

Evaluate the model
 **First Step**
 
## Prepare the Data

Imports and data loading
Import packages and libraries needed to compute descriptive statistics and conduct a hypothesis test.

**HERE MAKE SCREEN SHOOT FOR ALL IMPORTANT PACKAGE**  
# Packages for numerics + dataframes
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for Logistic Regression & Confusion Matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
**Import the data**

![Screenshot (228)](https://github.com/user-attachments/assets/c3c78703-026f-4c8b-ba42-e18874e734a0)

 ## Exploratory data analysis (EDA)##

 Start with head()
 
![Screenshot (230)](https://github.com/user-attachments/assets/43d53fac-69ba-4fbf-a156-e8e2da886675)

 shape and info().

![Screenshot (229)](https://github.com/user-attachments/assets/34f0ffb5-d5cd-4d66-86ee-cd682316a591)

The dataset has 700 missing values in the label column.


Now, check the class balance of the dependent (target) variable, label.

![Screenshot (233)](https://github.com/user-attachments/assets/3236ffd4-bcd0-40d5-9a71-784383ea39df)


![Screenshot (234)](https://github.com/user-attachments/assets/1b0df617-6b4e-4e23-80df-6246df641eb2)

The following columns all seem to have outliers assessing at the quartile values, standard deviation, and max values:

sessions
drives
total_sessions
total_navigations_fav1
total_navigations_fav2
driven_km_drives
duration_minutes_drives
All of these columns have max values that are multiple standard deviations above the 75th percentile. This could indicate outliers in these variables.




## PROSSES THE DATA

SInce we are interested in user churn, the label column is essential. Besides label, variables that tie to user behaviors will be the most applicable. All variables tie to user behavior except ID.
ID can be dropped from the analysis since we are not interested in identifying a particular user. ID does not provide meaningful information about the churn 

Use the drop() method to remove the ID column .

![Screenshot (232)](https://github.com/user-attachments/assets/054d94ec-46ed-4331-8aff-42aa594c3e55)



You know from earlier EDA that churn rate correlates with distance driven per driving day in the last month. It might be helpful to engineer a feature that captures this information.

Create a new column in df called km_per_driving_day, which represents the mean distance driven per driving day for each user..

![Screenshot (235)](https://github.com/user-attachments/assets/46f93b4a-9519-467f-a447-8f0b3f97e53e)

Note that some values are infinite. This is the result of there being values of zero in the driving_days column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.

Convert these values from infinity to zero. We can use np.inf to refer to a value of infinity.

Call describe() on the km_per_driving_day column to verify that it worked.

## Visulaization.

Begin by examining the spread and distribution of important variables using box plots and histograms.

**sessions**

**The number of occurrences of a user opening the app during the month**

![Screenshot (201)](https://github.com/user-attachments/assets/7b5c0538-299a-4ce2-a67c-954dea8ba116)

The sessions variable is a right-skewed distribution with half of the observations having 56 or fewer sessions. However,

as indicated by the boxplot, some users have more than 700.

![Screenshot (202)](https://github.com/user-attachments/assets/c1fe15e4-ba41-4248-bb2a-eb2321e57416)

An occurrence of driving at least 1 km during the month

![Screenshot (203)](https://github.com/user-attachments/assets/7d516d9d-19e1-4642-9e39-fc5c1a03f85c)

drived histogram

![Screenshot (205)](https://github.com/user-attachments/assets/97774a4e-0c0d-4d14-8fb3-efd21ac0155e)

The drives information follows a distribution similar to the sessions variable. It is right-skewed, approximately log-normal, with a median of 48. However, some drivers had over 400 drives in the last month.

A model estimate of the total number of sessions since a user has onboarded

![Screenshot (206)](https://github.com/user-attachments/assets/a5d7bd82-e00f-4875-98e0-d27c5c61cf60)


The total_sessions is a right-skewed distribution. The median total number of sessions is 159.6. 
This is interesting information because, if the median number of sessions in the last month was 56 and the 
Median total sessions was ~160, then it seems that a large proportion of a user's (estimated) total drives
Might have taken place in the last month. This is something you can examine more closely later.

![Screenshot (207)](https://github.com/user-attachments/assets/c4f4d4f7-92ad-418b-a1d8-9f8c629f4887)


The number of days since a user signed up for the app

![Screenshot (208)](https://github.com/user-attachments/assets/69668cc9-6723-40fe-9d22-d2aa0d877f7c)

Hstogram n_days_after_onboarding

![Screenshot (210)](https://github.com/user-attachments/assets/41a1d55f-a92b-46f9-96c2-163a63983f5d)

driven_km_drives
Total kilometers driven during the month

![Screenshot (211)](https://github.com/user-attachments/assets/9d660c0f-1625-4fc7-a950-75d9098f4a11)




![Screenshot (212)](https://github.com/user-attachments/assets/1620f4d5-de59-4ddc-9f33-c0e564d57e4e)

The number of drives driven in the last month per user is a right-skewed distribution with half the users driving under 3,495 kilometers. 
As you discovered in the analysis from the previous course, the users in this dataset drive a lot. 
The longest distance driven in the month was over half the circumferene of the earth.

duration_minutes_drives
Total duration driven in minutes during the month

![Screenshot (213)](https://github.com/user-attachments/assets/b993f4b0-6af7-4586-a599-dfef1e407f38)


![Screenshot (214)](https://github.com/user-attachments/assets/28d3b754-ec5a-43e3-baef-86b01ec07ac3)


The duration_minutes_drives variable has a heavily skewed right tail. Half of the users drove less than ~1,478 minutes (~25 hours),

But some users clocked over 250 hours over the month.

activity_days
Number of days the user opens the app during the month

![Screenshot (217)](https://github.com/user-attachments/assets/6d053b56-6dfc-4d04-b081-43474c437c6d)


![Screenshot (218)](https://github.com/user-attachments/assets/98c1c3f3-3b7d-4ced-a0c9-4232c5893173)



Within the last month, users opened the app a median of 16 times. The box plot reveals a centered distribution. The histogram shows a nearly uniform distribution of ~500 people opening the app on each count of days. However, there are ~250 people who didn't open the app at all and ~250 people who opened the app every day of the month.

This distribution is noteworthy because it does not mirror the sessions distribution, which you might think would be closely correlated with activity_days.


driving_days
Number of days the user drives (at least 1 km) during the month

![Screenshot (215)](https://github.com/user-attachments/assets/0e077523-ad33-47e8-bcb6-a54a5e2fd361)


![Screenshot (216)](https://github.com/user-attachments/assets/fb6566be-1bdc-40c9-82e9-d9889bdb00e3)

The number of days users drove each month is almost uniform, and it largely correlates with the number of days they opened the app that month, except the driving_days distribution tails off on the right.

However, there were almost twice as many users (~1,000 vs. ~550) who did not drive at all during the month. This might seem counterintuitive when considered together with the information from activity_days. That variable had ~500 users opening the app on each of most of the day counts, but there were only ~250 users who did not open the app at all during the month and ~250 users who opened the app every day. Flag this for further investigation later.

device
The type of device a user starts a session with

This is a categorical variable, so you do not plot a box plot for it. A good plot for a binary categorical variable is a pie chart.

![Screenshot (219)](https://github.com/user-attachments/assets/805cced7-359c-44e9-9fe6-ce2a333ad83d)

There are nearly twice as many iPhone users as Android users represented in this data.


label

Binary target variable (“retained” vs “churned”) for if a user has churned anytime during the course of the month

This is also a categorical variable, and as such would not be plotted as a box plot. Plot a pie chart instead.


![Screenshot (220)](https://github.com/user-attachments/assets/914857aa-4679-4540-b9ad-08ad26966a70)

Less than 18% of the users churned


driving_days vs. activity_days
Because both driving_days and activity_days represent counts of days over a month and they're also closely related, you can plot them together on a single histogram. This will help to better understand how they relate to each other without having to scroll back and forth comparing histograms in two different places.

Plot a histogram that, for each day, has a bar representing the counts of driving_days and user_days.

![Screenshot (222)](https://github.com/user-attachments/assets/729e9ea3-f43f-42da-9207-445041024879)

As observed previously, this might seem counterintuitive. After all, why are there fewer people who didn't use the app at all during the month and more people who didn't drive at all during the month?

On the other hand, it could just be illustrative of the fact that, while these variables are related to each other, they're not the same. People probably just open the app more than they use the app to drive—perhaps to check drive times or route information, to update settings, or even just by mistake.

Nonetheless, it might be worthwile to contact the data team at Waze to get more information about this, especially because it seems that the number of days in the month is not the same between variables.

Confirm the maximum number of days for each variable—driving_days and activity_days.

![Screenshot (223)](https://github.com/user-attachments/assets/dbcf7c73-8e2b-46b4-84fa-317cec506a99)

It's true. Although it's possible that not a single user drove all 31 days of the month, it's highly unlikely, considering there are 15,000 people represented in the dataset.

One other way to check the validity of these variables is to plot a simple scatter plot with the x-axis representing one variable and the y-axis representing the other.


![Screenshot (224)](https://github.com/user-attachments/assets/b0111a12-1359-4aef-89cb-8dc542d686ce)

Notice that there is a theoretical limit. If you use the app to drive, then by definition it must count as a day-use as well. In other words, you cannot have more drive-days than activity-days. None of the samples in this data violate this rule, which is good.

Retention by device

Plot a histogram that has four bars—one for each device-label combination—to show how many iPhone users were retained/churned and how many Android users were retained/churned.

![Screenshot (225)](https://github.com/user-attachments/assets/32bffbb5-0d59-40eb-8074-6df38aab3bb0)

The proportion of churned users to retained users is consistent between device types.

Retention by kilometers driven per driving day
In the previous course, you discovered that the median distance driven per driving day last month for users who churned was 697.54 km, versus 289.55 km for people who did not churn. Examine this further.

Create a new column in df called km_per_driving_day, which represents the mean distance driven per driving day for each user.

Call the describe() method on the new column.

![Screenshot (226)](https://github.com/user-attachments/assets/868e5442-ccf4-4c4a-bab5-b327cca989ac)

What do you notice? The mean value is infinity, the standard deviation is NaN, and the max value is infinity. Why do you think this is?

This is the result of there being values of zero in the driving_days column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.

Convert these values from infinity to zero. You can use np.inf to refer to a value of infinity.

Call describe() on the km_per_driving_day column to verify that it worked.
#### **Conclusion**

Analysis revealed that the overall churn rate is \~17%, and that this rate is consistent between iPhone users and Android users.

Perhaps you feel that the more deeply you explore the data, the more questions arise. This is not uncommon! In this case, it's worth asking the Waze data team why so many users used the app so much in just the last month.

Also, EDA has revealed that users who drive very long distances on their driving days are _more_ likely to churn, but users who drive more often are _less_ likely to churn. The reason for this discrepancy is an opportunity for further investigation, and it would be something else to ask the Waze data team about.

*****************************************************************************************************************************************************************************************
 ## Statistical Methods to Analyze and Interpret our Data.
 
In particular, We wants to know if there is a statistically significant difference in mean amount of rides between iPhone® users and Android™ users.
To do that we conduct a two-sample hypothesis test (t-test) to analyze the difference in the mean amount of rides between tpw group

**Research question**:

"Do drivers who open the application using an iPhone have the same number of drives on average as drivers who use Android devices?".

In order to perform this analysis, imust turn each label into an integer. The following code assigns a 1 for an iPhone user and a 2 for Android. It assigns this label back to the variable device_type.

You are interested in the relationship between device type and the number of drives. One approach is to look at the average number of drives for each device type. Calculate these averages.

device_type
1    67.859078
2    66.231838
Name: drives, dtype: float64

Based on the averages shown, it appears that drivers who use an iPhone device n have a higher number of drives on average. However, this difference might arise from random sampling, 

rather than being a true difference in the number of drives. To assess whether the difference is statistically significant,i can conduct a hypothesis test.

**Hypotheses:**

H0
 : There is no difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.

HA
 : There is a difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.
 
![Screenshot (227)](https://github.com/user-attachments/assets/801fec8f-7120-445c-acdd-3d502253bfdd)

*Since the p-value is larger than the chosen significance level (5%), you fail to reject the null hypothesis. You conclude that there is **not** a statistically significant difference* *in the average number of drives between drivers who use iPhones and drivers who use Androids*

business insight(s)  drawn from the result of your hypothesis test

The key business insight is that drivers who use iPhone devices on average have a similar number of drives as those who use Androids.


******************************************************************************************************************************************************************


## Regression modeling

In this section of projecti will build a binomial logistic regression model.

The goal is to build a binomial logistic regression model and evaluate the model's performance.

Outliers and extreme data values can significantly impact logistic regression models. After visualizing data, make a plan for addressing outliers by dropping rows, substituting extreme data with average data, and/or removing data values greater than 3 standard deviations.

EDA activities also include identifying missing data to help the analyst make decisions on their exclusion or inclusion by substituting values with dataset means, medians, and other similar methods.

Additionally, it can be useful to create variables by multiplying variables together or calculating the ratio between two variables. For example, in this dataset you can create a drives_sessions_ratio variable by dividing drives by sessions.
Use the drop() method to remove the ID column since you don't need this information for your analysis.

df = df.drop('ID', axis=1)

**FIRST**
Now, check the class balance of the dependent (target) variable, label.

df['label'].value_counts(normalize=True)
retained    0.822645
churned     0.177355
Name: label, dtype: float64
Call describe() on the data.
AFTRE DESCRIBE DATA WE CONCLODE 
Are there any variables that could potentially have outliers just by assessing at the quartile values, standard deviation, and max values?

Yes, the following columns all seem to have outliers:

sessions
drives
total_sessions
total_navigations_fav1
total_navigations_fav2
driven_km_drives
duration_minutes_drives
All of these columns have max values that are multiple standard deviations above the 75th percentile. This could indicate outliers in these variables.
Create a new column in df called km_per_driving_day, which represents the mean distance driven per driving day for each user.

Call the describe() method on the new column.

professional_driver
Create a new, binary feature called professional_driver that is a 1 for users who had 60 or more drives and drove on 15+ days in the last month.

Note: The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

To create this column, use the np.where() function. This function accepts as arguments:

A condition
What to return when the condition is true
What to return when the condition is false







































*******************************************************************************************************************************************************************************************

I have learned ....

* There is missing data in the user churn label, so we might need  further data processing before further analysis.
* There are many outlying observations for drives, so we might consider a variable transformation to stabilize the variation.
* The number of drives and the number of sessions are both strongly correlated, so they might provide redundant information when we incorporate both in a model.
* On average, retained users have fewer drives than churned users.

My other questions are ....

* How does the missingness in the user churn label arise?
* Who are the users with an extremely large number of drives? Are they ridesharing drivers or commercial drivers?
* Why do retained users have fewer drives than churned users? Is it because churned users have a longer history of using the Waze app?
* What is the user demographic for retained users and churned users?

My client would likely want to know ...

* What are the key variables associated with user churn?
* Can we implement policies to reduce user churn?

* What types of distributions did you notice in the variables? What did this tell you about the data?
Nearly all the variables were either very right-skewed or uniformly distributed. For the right-skewed distributions, this means that most users had values in the lower end of the range for that variable. For the uniform distributions, this means that users were generally equally likely to have values anywhere within the range for that variable.

Was there anything that led you to believe the data was erroneous or problematic in any way?
Most of the data was not problematic, and there was no indication that any single variable was completely wrong. However, several variables had highly improbable or perhaps even impossible outlying values, such as driven_km_drives. Some of the monthly variables also might be problematic, such as activity_days and driving_days, because one has a max value of 31 while the other has a max value of 30, indicating that data collection might not have occurred in the same month for both of these variables.

Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?
Yes. I'd want to ask the Waze data team to confirm that the monthly variables were collected during the same month, given the fact that some have max values of 30 days while others have 31 days. I'd also want to learn why so many long-time users suddenly started using the app so much in just the last month. Was there anything that changed in the last month that might prompt this kind of behavior?

What percentage of users churned and what percentage were retained?
Less than 18% of users churned, and ~82% were retained.

What factors correlated with user churn? How?
Distance driven per driving day had a positive correlation with user churn. The farther a user drove on each driving day, the more likely they were to churn. On the other hand, number of driving days had a negative correlation with churn. Users who drove more days of the last month were less likely to churn.

Did newer uses have greater representation in this dataset than users with longer tenure? How do you know?
No. Users of all tenures from brand new to ~10 years were relatively evenly represented in the data. This is borne out by the histogram for n_days_after_onboarding, which reveals a uniform distribution for this variable.









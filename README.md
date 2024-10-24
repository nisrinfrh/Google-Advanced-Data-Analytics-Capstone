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

## Prepare the Data

Prepare by reading in the data, viewing the data dictionary, and exploring the dataset to identify key variables for the stakeholder.

Start by importing the packages that i will need to load and explore the dataset

**Summary information**
View and inspect summary information about the dataframe by coding the following:


Does the dataset have any missing values?
**PICTURE1**
The variables label and device are of type object; total_sessions, driven_km_drives, and duration_minutes_drives are of type float64; the rest of the variables are of type int64. There are 14,999 rows and 13 columns.

The dataset has 700 missing values in the label column.

## PROSSES THE DATA

SInce we are interested in user churn, the label column is essential. Besides label, variables that tie to user behaviors will be the most applicable. All variables tie to user behavior except ID.
ID can be dropped from the analysis since we are not interested in identifying a particular user. ID does not provide meaningful information about the churn 

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


The total_sessions is a right-skewed distribution. The median total number of sessions is 159.6. This is interesting information because, if the median number of sessions in the last month was 56 and the median total sessions was ~160, then it seems that a large proportion of a user's (estimated) total drives might have taken place in the last month. This is something you can examine more closely later.

n_days_after_onboarding
The number of days since a user signed up for the app

# Calculate % of iPhone users and Android users in full dataset
df['device'].value_counts(normalize=True)
iPhone     0.644843
Android    0.355157
Name: device, dtype: float64
The percentage of missing values by each device is consistent with their representation in the data overall.


# Calculate counts of churned vs. retained
print(df['label'].value_counts())
print()
print(df['label'].value_counts(normalize=True))
retained    11763
churned      2536
Name: label, dtype: int64

retained    0.822645
churned     0.177355
Name: label, dtype: float64
This dataset contains 82% retained users and 18% churned users.

Next, compare the medians of each variable for churned and retained users. The reason for calculating the median and not the mean is that you don't want outliers to unduly affect the portrayal of a typical user. Notice, for example, that the maximum value in the driven_km_drives column is 21,183 km. That's more than half the circumference of the earth!

# Calculate median values of all columns for churned and retained users
df.groupby('label').median(numeric_only=True)
ID	sessions	drives	total_sessions	n_days_after_onboarding	total_navigations_fav1	total_navigations_fav2	driven_km_drives	duration_minutes_drives	activity_days	driving_days
label											
churned	7477.5	59.0	50.0	164.339042	1321.0	84.5	11.0	3652.655666	1607.183785	8.0	6.0
retained	7509.0	56.0	47.0	157.586756	1843.0	68.0	9.0	3464.684614	1458.046141	17.0	14.0
This offers an interesting snapshot of the two groups, churned vs. retained:

Users who churned averaged ~3 more drives in the last month than retained users, but retained users used the app on over twice as many days as churned users in the same time period.

The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median retained user.

It seems that churned users had more drives in fewer days, and their trips were farther and longer in duration. Perhaps this is suggestive of a user profile. Continue exploring!
Calculate the median kilometers per drive in the last month for both retained and churned users.

Begin by dividing the driven_km_drives column by the drives column. Then, group the results by churned/retained and calculate the median km/drive of each group.

# Add a column to df called `km_per_drive`
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
​
# Group by `label`, calculate the median, and isolate for km per drive
median_km_per_drive = df.groupby('label').median(numeric_only=True)[['km_per_drive']]
median_km_per_drive
km_per_drive
label	
churned	74.109416
retained	75.014702
The median retained user drove about one more kilometer per drive than the median churned user. How many kilometers per driving day was this?

To calculate this statistic, repeat the steps above using driving_days instead of drives.
# Add a column to df called `km_per_driving_day`
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# Group by `label`, calculate the median, and isolate for km per driving day
median_km_per_driving_day = df.groupby('label').median(numeric_only=True)[['km_per_driving_day']]
median_km_per_driving_day
	km_per_driving_day
label	
churned	697.541999
retained	289.549333
Now calculate the median number of drives per driving day for each group.

# Add a column to df called `drives_per_driving_day`
df['drives_per_driving_day'] = df['drives'] / df['driving_days']
​
# Group by `label`, calculate the median, and isolate for drives per driving day
median_drives_per_driving_day = df.groupby('label').median(numeric_only=True)[['drives_per_driving_day']]
median_drives_per_driving_day
The median user who churned drove 698 kilometers each day they drove last month, which is ~240% the per-drive-day distance of retained users. The median churned user had a similarly disproporionate number of drives per drive day compared to retained users.

It is clear from these figures that, regardless of whether a user churned or not, the users represented in this data are serious drivers! It would probably be safe to assume that this data does not represent typical drivers at large. Perhaps the data—and in particular the sample of churned users—contains a high proportion of long-haul truckers.

In consideration of how much these users drive, it would be worthwhile to recommend to Waze that they gather more data on these super-drivers. It's possible that the reason for their driving so much is also the reason why the Waze app does not meet their specific set of needs, which may differ from the needs of a more typical driver, such as a commuter.

Finally, examine whether there is an imbalance in how many users churned by device type.

Begin by getting the overall counts of each device type for each group, churned and retained.
 For each label, calculate the number of Android users and iPhone users
df.groupby(['label', 'device']).size()
label     device 
churned   Android     891
          iPhone     1645
retained  Android    4183
          iPhone     7580
dtype: int64
Now, within each group, churned and retained, calculate what percent was Android and what percent was iPhone.

# For each label, calculate the percentage of Android users and iPhone users
df.groupby('label')['device'].value_counts(normalize=True)
label     device 
churned   iPhone     0.648659
          Android    0.351341
retained  iPhone     0.644393
          Android    0.355607
Name: device, dtype: float64
The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and those ratios are both consistent with the ratio found in the overall dataset.
PACE: Execute
Questions:

Did the data contain any missing values? How many, and which variables were affected? Was there a pattern to the missing data?
The dataset has 700 missing values in the label column. There was no obvious pattern to the missing values.

What is a benefit of using the median value of a sample instead of the mean?
Mean is subject to the influence of outliers, while the median represents the middle value of the distribution regardless of any outlying values.

Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?
Yes. For example, the median user who churned drove 698 kilometers each day they drove last month, which is about 240% the per-drive-day distance of retained users. It would be helpful to know how this data was collected and if it represents a non-random sample of users.

What percentage of the users in the dataset were Android users and what percentage were iPhone users?
Android users comprised approximately 36% of the sample, while iPhone users made up about 64%

What were some distinguishing characteristics of users who churned vs. users who were retained?
Generally, users who churned drove farther and longer in fewer days than retained users. They also used the app about half as many times as retained users over the same period.

Was there an appreciable difference in churn rate between iPhone users vs. Android users?
No. The churn rate for both iPhone and Android users was within one percentage point of each other. There is nothing suggestive of churn being correlated with device.
## EDA ANALYSIS
**ID can be dropped from the analysis since we are not interested in identifying a particular user. 

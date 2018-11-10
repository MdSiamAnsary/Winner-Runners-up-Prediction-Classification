# Project
Winner Runners up Prediction Classification

# Used Tools
  - Windows 10 Pro
  - Python 3.6.4
  - JetBrains PyCharm 3.4.4
  - Anaconda Cloud
  
# Installed Packages 
| Package | pip download command  |
| ------ | ------ |
| pandas | pip install pandas |
| sklearn | pip install -U scikit-learn |
| warning |pip install pytest-warnings |
| astropy | pip install astropy |
| numpy | pip install numpy |


# Problem description
As football is a very popular worldwide sport, this work resolves around the different football leagues around the world. It is a classifying problem of predicting the Winners or Runners Ups of a league in a particular season.

# Dataset
The dataset of this problem has been created with the information from the site "World Football" (http://www.worldfootball.com). There are 120 entries in the dataset with fourteen columns. We use the records of four popular League records. They are:
* Premier league
* Bundesliga
* Ligue 1
* La Liga

The columns of the dataset are as such:
* YearFrom : The year a season starts
* YearTo : The year a season ends
* Club : Name of the football club
* Country : The country the club is from
* League : The league of participation
* Pld : Number of games played in the particular season
* W : Number of games won in the particular season
* D : Number of games drawn in the particular season
* L : Number of games lost in the particular season
* GF : Goals scored the club
* GA : Goals scored against the club
* GD : Goal difference
* Points : Points gained by the club in the particular season
* Outcome : If the club was winner or runner up that season

The target column is Outcome.

# Used Classification Models
* Logistic regression
* SGD Classification
* K Nearest Neighbors Classification
* Decision Tree Classification
* Ada Boost Classification
* Random Forest Classification

# Used Performance Metric Score
* Accuracy score
* Precision score
* Recall score
* F1 score

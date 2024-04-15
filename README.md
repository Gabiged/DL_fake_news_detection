# Fake News Detection Model
Use files in this order:
1. 01_DL_EDA_ML_models.ipynb;
2. 02_DL_distilbert.ipynb;
3. utils.py file is for functions used in this project.

The task for this project is to build a model to detect fake information. Using articles headlines and body text I will try to predict whether a particular article is fake or not.
For this work I will use 3 ML models (Logistic Regression, Random Forest and XGBoost) and DL model using pretrained Distilbert model.
<br>
The dataset source is: https://drive.google.com/file/d/1CZzfZDvE5E7HaHjk9yeyZDKil4_jkass/view?usp=drive_link<br>
ML part:
* Overall best results was achieved using Logistic Regression model;
* Additionally created features were not so useful therefore only "text" feature was used for prediction;
* After filtering top 10 most popular tokens results were significantly worse, however, the aim was to develop a model that is able to best generalise the given text by distinguishing between fake and real news. With this task best result was achieved using Logistic REgression model.<br>
DL part:
* The main issue has been the database itself. One part was the real news as articles published by Reuters. In principle, these are articles that have been edited several times and have almost lost their individuality, which is quite important for textual classification. In this case, I have removed all references to this news agency in order to reduce the bias. The second part of the database (of fake news) is the statements and comments in the social network (I can guess from the general structure of the statements). This part is highly individualized and linguistically it would be easy to predict authorship. Overall, therefore, I could conclude that these two dataset parts are hardly comparable.
* In order to eliminate bias, I have tried to remove certain items (special characters). At the same time I have tried to keep a few most common features in fake news: upper cased words, exclamation marks and multiple dots. Linguistically, these items are mostly used to give special emphasis to the content, and since the aim is to identify fake news, I think this choice has justified the effort.
* Overall tuned model gave better results compared with ML Logistic regression model if we compare False Negative results. Logistic Regresion gave 560 false news predicted as real ones and DL model gave 127 false news as negatives.

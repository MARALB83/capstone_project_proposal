# Machine Learning Engineer Nanodegree
## Capstone Proposal
Mario Albuquerque  
May 9th, 2018

## Proposal

### Domain Background

<p style='text-align: justify;'>The subject of the project was taken from a Kaggle competition named ["Avito Demand Prediction Challenge"](https://www.kaggle.com/c/avito-demand-prediction). Today's retail environment is increasingly dominated by online platforms (eg Amazon, Alibaba, etc.) that want to reach as wider audience as possible in order to sell goods and services. In this business it is crucial to determine how the product is advertised so that the right balance between maximizing demand (marketing spending) and profit (pricing optimization) can be achieved.  
My interest in this particular project is to combine text, numerical, and imaging data in one data science challenge so that a more expanded set of my skills can be showcased.</p>

### Problem Statement

<p style='text-align: justify;'>The problem is one of determining the demand for an online advertisement given its endogenous characteristics (eg. advertisement title, item price, type of advertiser, and advertisement image quality), geographical location (region and city where the item is located), similarity to other items (category and parent category of the advertised item), and historical demand in comparable scenarios (eg. cumulative ads posted by an user and item price distance to the category mean). These are the features, or inputs for the problem.  
The demand for the ad is expressed as a probability (named *deal_probability* in the dataset) and this is the target variable or the output of the model. Without any further transformation, the problem is a supervised regression problem as the output variable is continuous (probabilities). However, it will also be attempted to transform the output variable (*deal_probability*) into a 2-label supervised classification problem, where the model will assign *Unlikely* and *Likely* labels depending on whether the probability of the deal is below 50% or not, respectively.</p>

### Datasets and Inputs

<p style='text-align: justify;'>The dataset was provided by Avito (through Kaggle) and it represents classified advertisements along with the images (when available) of the products/services being advertised:</p>

* **_train.csv_**: Contains 1,503,424 ads with the following columns:  
![column_desc](./Images/columns_desc.PNG?raw=true)  
Source: https://www.kaggle.com/c/avito-demand-prediction/data

<p style='text-align: justify;'>The approach taken for this capstone project is to further divide the _train.csv_ file into train and testing subsets. Although Avito provides additional records assigned to the testing subset in the Kaggle challenge, they do not have the target/dependent variable that would make it possible to evaluate machine learning models. The Kaggle challenge sponsor holds back the *deal_probability* column from the testing group so that it remains a competitive and truly out-of-sample exercise.</p>

* **_train_jpg_**:<p style='text-align: justify;'>A folder containing 1,390,836 images that correspond to the classified ads in the **_train.csv_** file that had an image. The column named *image* in the *train.csv* file has the filename of the image associated with a specific ad.</p>

The data can be downloaded in this website: https://www.kaggle.com/c/avito-demand-prediction/data.  
The **_train.csv_** file is named **_train.csv.zip_**, and the **_train_jpg_** folder with images is named **_train_jpg.zip_**.</p>

### Solution Statement

The solution for this problem will take 2 approaches:

* **A classification-based approach**: <p style='text-align: justify;'>Where the dependent variable is going to be discretized into 2 labels based on having a _deal_probability_ below 50%, or not. The labels will be named *Likely* and *Unlikely*, based on the *deal_probability* value being equal or above 50%, and below 50%, respectively.  
In this approach, multiple models (eg. Naive Bayes, Logistic Regression, SVM, Random Forests, etc.) will be tried and combined so that the F1-Score is maximized in the validation set. The final performance evaluation will be assessed in the testing subset.</p>

* **A regression-based approach**: <p style='text-align: justify;'>Where the dependent variable is going to be predicted through multiple regression models.   
In this approach, multiple models (eg. linear regression, Lasso regression, Ridge regression, ElasticNet regression, etc.) will be tried and combined so that the Root Mean Squared Error is minimized in the validation set. The final performance evaluation will be assessed in the testing subset.</p>

### Benchmark Model

<p style='text-align: justify;'>The benchmark models were developed by selecting the most intuitively significant feature from the available columns in the *train.csv* file. That feature is *price* as it is, intuitively, an important driver of deal probability. The higher the price, the lower the deal probability, and vice-versa.</p>

* **Classification-based benchmark**: <p style='text-align: justify;'>For each item category, compute whether the price of a specific ad is above or below the median. If it is equal or above, then the predicted label is "Unlikely", if it is below the category median, then the predicted label is "Likely". Notice that it is important to condition by item category, otherwise big ticket items like cars would be systematically flagged as "Unlikely", even though they could be excellent deals.</p>

* **Regression-based benchmark**:<p style='text-align: justify;'>A linear regression with *price* as the independent variable and *deal_probability* as the dependent variable. The parameters of the model are going to be estimated across the full dataset to predict *deal_probability*.</p>

### Evaluation Metrics

As both classification and regression-based approaches are going to be tested, there is going to be one metric for each problem type.

* **For classification-based models**: The evaluation metric is the F1-Score as it balances precision and recall. The equation is given by:  
![f1_score](./Images/f1_score.PNG?raw=true)   
Source: https://en.wikipedia.org/wiki/F1_score

* **For regression-based models**: <p style='text-align: justify;'>The evaluation metric is the root mean squared error (RMSE) which is a quantitative way to express the average deviation of the predicted deal probability from the actual value. The equation is given by:</p>
![rmse](./Images/rmse.PNG?raw=true)  
Source: https://www.kaggle.com/c/avito-demand-prediction#evaluation

### Project Design
<p style='text-align: justify;'>
At a high level, the project workflow can be depicted in the following diagram:</p>

![workflow](./Images/workflow.png?raw=true)  

* **I. Exploratory data analysis**: <p style='text-align: justify;'>distribution of target variable (*deal_probability*); distribution of available features; relationship between available features and target variable.</p>
* **II. Feature engineering**: <p style='text-align: justify;'>translation of numerical data into the same magnitude (ZScoring) vs. original values; natural language processing of text data rangin from simple word/punctuation count to feature extraction through Tf-Idf and dimensionality reduction; processing of imaging data to identify unambiguous high-quality pictures using neural networks.</p>
* **III. Model development***: <p style='text-align: justify;'>train/test data split; model training and tuning through parameter grid search (train vs. validation datasets); different models tested (classification: Naive Bayes, Logistic Regression, SVM, etc.; regression: linear regression, Lasso regression, Ridge regression, etc.); model evaluation (classification: F1-Score and regression: RMSE).</p>
* **IV. Final solution**: <p style='text-align: justify;'>Best model maximizes F1-Score in testing subset for classification; minimizes RMSE in testing subset for regression. Check ensembling techniques.</p>
### References

Dataset: https://www.kaggle.com/c/avito-demand-prediction/data

F1-Score Evaluation Metric: https://en.wikipedia.org/wiki/F1_score

Kaggle Avito Demand Prediction Challenge: https://www.kaggle.com/c/avito-demand-prediction

Root Mean Squared Error Evaluation Metric: https://www.kaggle.com/c/avito-demand-prediction#evaluation
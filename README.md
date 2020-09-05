# Fake Reviews Detection

## Introduction

Online reviews provide customers and business owners a good evaluation metric for the target products or services. Therefore, knowing the authenticity of reviews is essential, because they may affect how we make a decision. In this work, we tackled the fake review detection problem by designing new features, re-sampling training data and ensembling different machine learning models. Our best performance XGBoost model achieved 0.404 Average Precision and 0.862 AUC score on test data.

{% include gallery caption="" %}

## Feature Engineering

#### 1. Linguistic Features

- **HTML Tags and URL:** Web-scraped data may contain HTML tags which are not useful in text data, so we use a Python library, BeautifulSoup, to remove them.
- **Accented characters:** Since we are dealing with English reviews, accented characters, e.g. café, can be confusing. In this project, we convert all accented characters into normal English characters, e.g. cafe.
- **Punctuation \& Special Characters:** The Punctuation marks, such as question marks and quotes, inclusive of any special characters are also removed from the text data
- **Stopwords:** In the Spark NLP pipeline, we use the pre-defined stopwords collection from NLTK to extract the words which do not add much value in the article.
- **Lemmatization \& Lower Case:** One word can be written in different form with different tenses. Lemmatization helps us unifying them into a single format. After that, we convert all tokens into the lower case.


#### 2. Review-centric Features
- Review length
- Absolute rating deviation from product’s average rating
- **Earlier time review:** Whether the current review is written within the T=70 months from the first review for this product. It's reasonable for a spammer to post a review as early as possible to increase their chances of being viewed.
- **Whether a review is written on holiday or not [new]:** It's more effective to write fake reviews on holidays, since people are more likely to go out on these days.
- Whether a review is written on weekend or not

#### 3. Reviewer-centric Features

- Number of reviews per user
- Max/avg reviews per user per day
- Percentage of positive/negative reviews per user
- **Burstiness:** Whether all reviews from a user are written within T=28 days.
- **Avg rating deviation:** How different are one's reviews from others.
- Avg/Var review length per user
- **Avg content similarity (n-gram):** Compute the average of pairwise jaccard similarities among user’s reviews. A review is represented as a 500d uni-gram vector.
- **Avg content similarity (GloVe):** Compute the average of pairwise cosine similarities among user’s reviews. A review is represented as a 50d GloVe vector. We argue that GloVe can better embed the meaning of the text into a smooth manifold. As a result, it can capture the information where uni-gram cannot.

    
## Dealing with Imbalanced Data

- **Random under-sampling (RUS):** Only a subset of samples from the majority class are used. Some important information in the out-of-sample data may be lost.
- **Random over-sampling (ROS):** Each sample in the minority class might be sampled for multiple times. However, replicating too much information may result in over-fitting.
- **SMOTE (Synthetic Minority Oversampling Technique):** Basically, the technique generates new samples by interpolating of its nearest neighbors.
- **ROS + RUS:** This method can benefit from the advantages of both approaches.
- **SMOTE + RUS:** For the model with only the linguistic features, we use this method to create a dataset with 50,000 in both of the target classes.
- **Ensemble of RUS:** Perform RUS several times, each time with a different subset of samples from the majority class. Make the final prediction by voting.

# What is website classification and why do we need it?

Website classification is broadly speaking the task of classifying a website into one or more categories.

The process of categorizing websites is usually automated, and often uses supervised machine learning models as the most effective solution.
Use cases

Website classification has many use cases from a wide range of fields. One important application is in the field of [business intelligence](https://www.tableau.com/learn/articles/business-intelligence). By analyzing the content and structure of your competitors’ websites, you can get an idea what they are up to, and discover new opportunities for your own business.

Another use case is in the management of website content. A company may have different policies for different types of websites, such as different levels of access based on the age range of visitors to the site and this can be inferred by the category of the website.

Another important application is cybersecurity, where we classify websites into potential spam, phishing or websites that we do not want to be visited by e.g. our employees.

This allows us to block such potentially harmful websites and prevent users from accessing them. We can also employ whitelists, where only safe websites on a pre-defined list are allowed to be accessed by users of the network.

Important use of website classification is also online advertising, where advertisers can target ads more effectively if they know what the content of the site is about.
[Real-time Bidding companies] (https://en.wikipedia.org/wiki/Real-time_bidding) use it to decide which advertiser is appropriate for given publisher. 

# Taxonomies

Taxonomies are developed to help categorize content and make it easier to find. In the context of websites, an ad-focused taxonomy is most useful and the [Internet Advertising Bureau IAB] (https://www.iab.com/) has developed a taxonomy geared toward ads and marketing.

This taxonomy, which can be found at the IAB website, is constantly being revised based on changing user behaviors and categories. For this reason, if you are using the IAB taxonomy to categorize websites, it is important that you use the latest version.

If your website is focused on ecommerce then a different, products oriented taxonomy may be more appropriate. The most well known ones in this segment are those from Google:
https://www.google.com/basepages/producttype/taxonomy.en-US.txt

Google product taxonomy is structured by product categories and subcategories, making it easy to structure your content. It has several levels of depth or "Tiers". There are more than 1000 (sub)categories in the taxonomy, so you'll most likely find the right one for your products.
Machine learning models

Before you start building a supervised machine learning model for automated website categorization, you need to prepare a large amount of high quality training data. The more training data in your data set and the better their quality in terms of relevance and diversity, the more accurate and reliable your model will be. Therefore, it is recommended that you invest most of your time and resources into this part of the process.

There are several ways to collect training data for [website classification](https://www.websitecategorizationapi.com/). One way is to use existing datasets from various agencies or other third parties. You can also use existing web-crawling tools to crawl websites yourself and collect their content into a dataset.
A useful tool to accomplish this is using [a reverse ip api](https://reverseiplookupapi.com/) to find appropriate domains. 

Another option is to manually curate a dataset by simply opening up websites that are relevant to your use case and categorising them according to your custom taxonomy or taxonomy from Google, Facebook or IAB.

# Article/Content extraction

Websites consist of a mix of content and supporting elements like menus, sidebars, and footers. These ancillary elements are usually less relevant to the central topic of the website; they may often be non-unique across multiple websites.

Consider a news website: the menus, footer, and headlines/teasers in the sidebar may be common across many articles on that website. What we are generally interested in is the content of the article. This is where article extraction comes into play.

Article extraction consists of extracting text from a website that is likely to be relevant to the topic of interest from a website that contains other material which is less related (or unrelated) to this topic.

Text pre-processing is an important part of data pipeline for website categorization models. As we are dealing with websites, the first part consists of extraction of relevant text from the websites. For most cases, we want to remove all non-article parts of web page as part of so-called article extraction.

There has been a lot of work done on the topic of content extraction. A great early research paper on this topic is https://www.researchgate.net/publication/221519989_Boilerplate_Detection_Using_Shallow_Text_Features. It has an open source implementation written in Java.

There are also many ready made libraries available for content extraction written in python which is more commonly used in data science, e.g. goose3 (https://github.com/goose3/goose3) and newspaper (https://github.com/codelucas/newspaper).

# Machine learning models for Website classification

Many machine learning models can be utilized for the purpose of classifying websites. 

## Multinomial Naïve Bayes (NB)

The Multinomial Naive Bayes algorithm is a classification algorithm that works by assigning probabilities to each possible outcome. The algorithm is designed for the case where the distribution of data falls into one of three or more categories. For example, if you are a marketing professional trying to predict whether an email will result in a sale, you would be interested in predicting whether an email will result in a click, a purchase, or neither of these things. If your data set includes information about what other products shoppers have purchased, this is called "multinomial" data because it has more than two outcomes.

Multinomial Naive Bayes assigns probabilities to each category based on the frequency with which each category appears in your data set; these probabilities are known as "priors."

Example implementation of Naive Bayes: [https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

## Logistic Regression (LR)

 Logistic regression is a machine learning model that is used to estimate values of a categorical response variable, given a set of predictor variables.

For example, if you want to predict whether a person is likely to be employed or unemployed, you would use logistic regression. It takes the input data and uses it to estimate the probability that an individual will be unemployed based on their age and gender.

Logistic regression works best when there are few features or predictors, but many observations in the training dataset.

More about logistic regression: [https://www.sciencedirect.com/topics/computer-science/logistic-regression](https://www.sciencedirect.com/topics/computer-science/logistic-regression)

## Support Vector Machines

 Support Vector Machines is a machine learning model that can be used to classify data. It can be used for both regression and classification problems. The goal of SVM is to find a hyperplane that separates the data into two classes of points, while maximizing the margin between these two classes. This process is called "maximizing the margin".

The training process consists of finding an optimal separating hyperplane by solving an optimization problem. The optimization problem has a linear objective function, which means that the hyperplane must be linear if it's going to fit our data perfectly.

Example implementation of SVM model: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)

## Stochastic Gradient Descent 

 Stochastic Gradient Descent (SGD) is a machine learning model that uses stochastic optimization to find the optimal parameters for a given machine learning model. It is one of the key algorithms used in [data science](http://www.datascienceconsultant.net/). 

The process begins by defining a cost function, which is a function that maps the predicted output to the actual value and calculates the difference between them. This function is then minimized using gradient descent, which is an iterative method that involves taking small steps towards the minimum value of this function.

During each iteration, SGD takes a step in a random direction and updates its parameters accordingly. It does this by calculating an error term for every parameter in its model and then adjusting those parameters so that they are closer to their optimum values.

## k-Nearest-Neighbors algorithm

 K-Nearest-Neighbors (k-NN) is a machine learning model that uses a distance metric to classify data into groups. The k-NN algorithm is used when the variables in question are continuous, but can also be used with categorical variables by converting their values into binary features.

The k-NN algorithm works by finding the "k" nearest neighbors of each data point in the training set, where "k" is usually between 3 and 10. For each neighbor, the k-NN algorithm calculates the distance between itself and its neighbor. Then it compares this value to the distance between itself and every other data point in its training set. k-NN are one of the more useful algorithms in [AI](http://www.aiconsultingservices.net/). 

If there are more than one neighbor with a smaller distance than itself, then it classifies itself as belonging to their group. If there are no neighbors with a smaller distance than itself, then it classifies itself as belonging to another group or as being unclassified (if all of its neighbors have larger distances).

More about KNN: [https://www.ibm.com/topics/knn](https://www.ibm.com/topics/knn)

## Random Forests

 A Random Forest machine learning model is a type of ensemble method. It's a classification algorithm that builds multiple decision trees from different subsets of the training data and then uses the majority vote to predict the class of new instances. The trees are generated by sampling with replacement, which means that some records will be selected more than once and some not at all. This is done to reduce bias and improve variance, which results in better overall performance on unseen data.

The resulting decision tree is pruned (in order to reduce overfitting), resulting in a set of decision nodes that represent individual predictions for each record in the test set. A random forest model can be used as either a classifier or a regression model by changing how it calculates probabilities or regression coefficients respectively. 

# Conclusion

Website classification is very important in [machine learning and natural language processing](https://www.machinelearningconsulting.net). It has many use cases in a number of industries, including Cybersecurity and Online Stores Categorizations.

An important part of website classification is the extraction of relevant text from websites (by removing boilerplate elements), where special machine learning models can be used for this purpose.

For text classification itself, a wide range of machine learning models can be used, from Logistic Regression to deep learning nets.

# Application of classification in finance

One of the most important applications of Mean CVaR methodology is in relation to asset allocation. It is well known from study of [Brinson, Beebower and Hood](https://www.jstor.org/stable/4478947) that the asset allocation can explain over 90% of variability of portfolio returns and thus represents the cornerstone of successful asset management. The standard approach to asset allocation usually assumes constant relative weights of asset classes through time and thus not adapts to changing risk regimes on capital markets, which is especially important in times of distress and extreme risk events and losses. One [asset allocation software solution](https://www.alpha-quantum.com/portfolio_optimisation.html) for this weakness of strategic asset allocation is to use dynamic asset allocation based on Mean CVaR methodology. By using this approach, we can form portfolios, which dynamically adjust to changing correlations between asset classes and shifts in expected returns, leading to portfolios with higher returns and lowering of risks.

 

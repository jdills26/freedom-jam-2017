# freedom-jam-2017
This three part analysis uses a dataset that is a collection of tweets from identified pro-ISIS twitter users.  

Part one includes an exploratory analysis of the dataset that uses the timestamp to visualize the frequency of the tweets over time, looks most popular locations, and introduces some natural language processing.  

Part two attempts to predict if a tweet is pro-ISIS or not by bringing in an additional dataset of random tweets that are not pro-ISIS.  It’s a binary classification problem that predicts the label of the tweet (ie. if it is from the pro-ISIS set or the additional set).  A train-test split is after combining these two datasets to train a Naïve Bayes algorithm.  CountVectorizer is also used to create a document term matrix that tokenizes and counts the occurrence of words in the training dataset (prior to training the learner).  

Part three is an acknowledgment to the rise in popularity of using deep learning with natural language processing – I adapt a tensorflow tutorial that uses a Word2Vec skip-gram model over pro-ISIS twitter data and visualizes the learned embeddings using t-SNE.

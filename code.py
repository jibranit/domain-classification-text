# --------------
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# path_train : location of test file
# Code starts here

df = pd.read_csv(path_train)
df.head()

def label_race(row):
    category = []
    n=1
    for i in range(len(row)):
        category.append(row[i:n].unstack()[row[i:n].unstack()=='T'].index[0][0])
        n+=1
    return category

category = label_race(df)

df['category'] = category
df=df[['message','category']]








# --------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# Sampling only 1000 samples of each category
df = df.groupby('category').apply(lambda x: x.sample(n=1000, random_state=0))

all_text = df['message']

X = TfidfVectorizer(stop_words = 'english').fit_transform(all_text)

y = LabelEncoder().fit_transform(df['category'])


# --------------
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Code starts here

X_train,X_val,y_train, y_val = train_test_split(X,y,test_size=0.3, random_state=42)

log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_val)

log_accuracy = accuracy_score(y_val,y_pred)
print(log_accuracy)

nb = MultinomialNB()

nb.fit(X_train,y_train)

y_pred_nb = nb.predict(X_val)

nb_accuracy = accuracy_score(y_val,y_pred_nb)
print(nb_accuracy)

lsvm = LinearSVC(random_state=0)

lsvm.fit(X_train,y_train)

y_pred_lsvm = lsvm.predict(X_val)

lsvm_accuracy = accuracy_score(y_val,y_pred_lsvm)
print(lsvm_accuracy)





# --------------
# path_test : Location of test data

#Loading the dataframe
df_test = pd.read_csv(path_test)

#Creating the new column category
df_test["category"] = label_race(df_test)

#Dropping the other columns
drop= ["food", "recharge", "support", "reminders", "nearby", "movies", "casual", "other", "travel"]
df_test=  df_test.drop(drop,1)

# Code starts here

all_text = df_test["message"].str.lower()

# Transforming using the tfidf object - tfidf
X_test = tfidf.transform(all_text).toarray()

# Transforming using label encoder object - le
y_test = le.transform(df_test["category"])

# Predicting using the logistic regression model - logreg
y_pred = log_reg.predict(X_test)
log_accuracy_2 = accuracy_score(y_test,y_pred)
print (str(log_accuracy_2)+(" is the accuracy of the logistic regression model"))

# Predicting using the naive bayes model - nb
y_pred = nb.predict(X_test)
nb_accuracy_2 = accuracy_score(y_test,y_pred)
print (str(nb_accuracy_2)+(" is the accuracy of the Naive Bayes model"))


# Predicting using the linear svm model - lsvm
y_pred = lsvm.predict(X_test)
lsvm_accuracy_2 = accuracy_score(y_test,y_pred)
print (str(lsvm_accuracy_2)+(" is the accuracy of the Support Vector model"))



# --------------
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
# import nltk
# nltk.download('wordnet')

# Creating a stopwords list
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Creating a list of documents from the complaints column
list_of_docs = df["message"].tolist()

# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]

# Code starts here

dictionary = corpora.Dictionary(doc_clean)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

lsimodel = LsiModel(corpus=doc_term_matrix, num_topics=5, id2word=dictionary)

pprint(lsimodel.print_topics())








# --------------
from gensim.models import LdaModel
from gensim.models import CoherenceModel

# doc_term_matrix - Word matrix created in the last task
# dictionary - Dictionary created in the last task

# Function to calculate coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    topic_list : No. of topics chosen
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    topic_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, random_state = 0, num_topics=num_topics, id2word = dictionary, iterations=10)
        topic_list.append(num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return topic_list, coherence_values


# Code starts here
topic_list,coherence_value_list = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=doc_clean, start=1, limit=41, step=5)

opt_topic = topic_list[coherence_value_list.index(max(coherence_value_list))]
print(opt_topic)

lda_model = LdaModel(corpus=doc_term_matrix, num_topics=opt_topic, id2word = dictionary, iterations=10 , passes=30, random_state=0)

pprint(lda_model.print_topics())



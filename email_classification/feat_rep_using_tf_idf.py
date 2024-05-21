from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#1. Data representation using TF-IDF
#TF-IDF score represents the relative importance of a term in the document and the entire corpus. 
#TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), 
#the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents 
#in the corpus divided by the number of documents where the specific term appears.


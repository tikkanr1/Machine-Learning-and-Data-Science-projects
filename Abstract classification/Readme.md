**Goal**  
Classification of abstracts of scientific publications using natural language processing (NLP) methods.

**Approach**
1. Data preprocessing (tokenization, stemming, lemmatization)
2. Transformation into tfidf (term frequency * inverse document frequency) form
3. Classification using two clustering methods:
  a) Basic K-means clustering.
  b) Scatter/Gather Buckshot clustering with ngrams.

**Results**  
The accuracy of the results were measured using normalized mutual information(NMI).
The results show that the advanced Scatter/Gather Buckshot method with ngrams produces a better NMI score (0.812)
than the basic K-means method (0.677) and is therefore more accurate.  

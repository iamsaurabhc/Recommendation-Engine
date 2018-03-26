# Libraries to use
from __future__ import division
import pandas as pd
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
from datasketch import MinHashLSHForest, MinHash
import math
from collections import Counter
from nltk import cluster
from sklearn import neighbors
from unidecode import unidecode


articles = pd.read_csv('data/articles.tsv',delimiter='\t',encoding='utf-8')
orders = pd.read_csv('data/orderlines.tsv',delimiter='\t',encoding='utf-8')
promoArticles = pd.read_csv('data/articles_available_for_promo.tsv',delimiter='\t',encoding='utf-8')

'''
print ("Articles:\n",articles.head())
print ("\nOrders\n:",orders.head())
print ("\nPromo Articles:\n",promoArticles.head())
'''
ordersWName = pd.merge(orders, articles, on=['article_id'])
promoArticlesWName = pd.merge(promoArticles, articles, on=['article_id'])

#print (ordersWName.head())

X = ordersWName[ordersWName['promo_type'].notnull()]
X = X.sample(frac=1)
X['promo_type'] = X['promo_type'].astype('category')
X['promo_type_cat'] = X['promo_type'].cat.codes
X[['promo_type','promo_type_cat']].head()

corpus = list(X['art_name'])
labels = list(X['promo_type_cat'])
labels = [ int(x) for x in labels ]

X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2)

vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
test_corpus_tf_idf = vectorizer.transform(X_test)

model1 = LinearSVC()
model2 = MultinomialNB()   
#model3 = neighbors.KNeighborsClassifier()

model1.fit(train_corpus_tf_idf,y_train)
model2.fit(train_corpus_tf_idf,y_train)
#model3.fit(train_corpus_tf_idf,y_train)

result1 = model1.predict(test_corpus_tf_idf)
result2 = model2.predict(test_corpus_tf_idf)
#result3 = model3.predict(test_corpus_tf_idf)

score1 = accuracy_score(result1, y_test)
score2 = accuracy_score(result2, y_test)
#score3 = accuracy_score(result3, y_test)

print (score1,score2)

def buildVector(iterable1, iterable2):
    counter1 = Counter(iterable1)
    counter2= Counter(iterable2)
    all_items = set(counter1.keys()).union( set(counter2.keys()) )
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2
'''
forest = MinHashLSHForest(num_perm=128)
for prod in list(set(promoArticlesWName['art_name'])):
    m1 = MinHash(num_perm=128)
    for p in prod.split():
        m1.update(unidecode(unicode(p.encode('utf-8'), encoding = "utf-8")))
        #m1.update(p.encode('utf8'))
    forest.add(unidecode(unicode(prod.encode('utf-8'), encoding = "utf-8")), m1)
    #forest.add(str(prod.encode('utf8')), m1)

forest.index()

'''
personalizedPromotions = pd.DataFrame()
count = 0
total = len(list(set(ordersWName['customer_id'])))

for user in list(set(ordersWName['customer_id'])):
    cust = ordersWName[ordersWName['customer_id'] == user]
    count+=1
    
    if len(list(cust['art_name'])) >1:
        #print ('Total orders:',user,'=',len(list((cust['art_name']))), '| \n%.2f'%(count/total)*100,' percent complete!')
        print (str(count)+' out of '+str(total)+' remaining')
        names = []
        alter = []

        for i in list(set(cust['art_name'])):
            for j in list(set(promoArticlesWName['art_name'])):
                #fuzz.ratio(i, j)
                if i!=j and j not in list(set(cust['art_name'])):
                    v1,v2 = buildVector(i.split(),j.split())
                    dist = cluster.util.cosine_distance(v1,v2)
                    if dist<0.3:
                        #print i,"==>",j
                        names.append(j)
        
        '''
        for p in list(set(cust['art_name'])):
            m2 = MinHash(num_perm=128)
            for _p in p.split():
                m2.update(unidecode(unicode(_p.encode('utf-8'), encoding = "utf-8")))
            result = forest.query(m2, 1)
            if result:
                for res in result:
                    names.append(str(res))
        '''
        '''
        #art_id = []
        for x in name:
            #print promoArticlesWName.loc[promoArticlesWName['art_name'] == unidecode(unicode(x.encode('utf-8'), encoding = "utf-8")), 'article_id']
            #art_id.append(str(promoArticlesWName.loc[promoArticlesWName['art_name_decode'] == unidecode(unicode(x.encode('utf-8'), encoding = "utf-8")), 'article_id'].iloc[0]))
            if x not in list(set(cust['art_name'])):
                names.append(x)
        '''
        #names = [x for x in name if x not in list(set(cust['art_name']))]

        if len(names)!=0:
            relevantProducts = vectorizer.transform(names)
            resultsCategorical = model1.predict(relevantProducts)
            #print len(names)
            results = []
            customer = []
            art_id = []
            for r in resultsCategorical:
                results.append(str(X.loc[X['promo_type_cat'] == r, 'promo_type'].iloc[0]))
                customer.append(str(cust['customer_id'].iloc[0]))
            res = {
                'customer_id':customer,
                #'article_id':art_id,
                'art_name':names,
                'promo_type':results
            }

            r = pd.DataFrame(res)
            r = r.drop_duplicates(subset=['customer_id','art_name'])
            personalizedPromotions = personalizedPromotions.append(r)
        else:
            print ('No similar products | Not Taking User',user,'=',len(list((cust['art_name']))))
    else:
        print ('Not Taking User',user,'=',len(list((cust['art_name']))))

    if count%1000==0:
        personalizedPromotions.to_csv('predict_part.csv', index=False)

print (personalizedPromotions.head())
personalizedPromotions.to_csv('personalizedPromotions.csv', index=False)
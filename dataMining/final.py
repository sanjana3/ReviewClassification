
# coding: utf-8

# In[109]:

import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
plt.switch_backend('SVG') 
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gender_predictor.GenderClassifier import classify_gender
from sklearn.decomposition import TruncatedSVD
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xlrd, xlwt
import xlsxwriter
from xlutils.copy import copy
from matplotlib.ticker import FuncFormatter


# In[24]:


def readData():    
    dout = open('review_data.pkl', 'rb')
    data1 = pickle.load(dout)
    data = pd.DataFrame()
    data['text'] = data1['reviewText']
    data['labels'] = data1['overall']
    dout.close()
    return data


def process():
    data = readData()

    data.loc[data['labels'] == 2 , 'labels'] = 1
    data.loc[data['labels'] == 3 , 'labels'] = 2
    data.loc[data['labels'] == 4 , 'labels'] = 3
    data.loc[data['labels'] == 5 , 'labels'] = 3

    df1 = data[data['labels'] ==1]
    df2 = data[data['labels'] ==2]
    df3 = data[data['labels'] ==3]

    r1 = df1.sample(frac=0.30, replace=False)
    r2 = df2.sample(frac=0.24, replace=False)
    r3 = df3.sample(frac=0.02, replace=False)

    frames = [r1, r2, r3]
    result = pd.concat(frames)
    return result
    

# data_sample = process()


# In[7]:


def _vectorizers(which, df, what, Cvec):
    
    if what == 'train':
        if which=='cv':
            
            cvec_train = Cvec.transform(df['text'])
            penkiDF = pd.DataFrame(cvec_train.toarray())
            penkiDF['labels'] = df['labels'].values
            return penkiDF

        if which=='tf':
            
            train_tf = tfidf_vec.transform(df['text'])
            penkiDF_tf = pd.DataFrame(train_tf.toarray())
            penkiDF_tf['labels'] = df['labels'].values
            return penkiDF_tf
        
    if what == 'test':
        if which=='cv':
            
            cvec_test = Cvec.transform(df['test'])
            testDF = pd.DataFrame(cvec_test.toarray())
            return testDF
        
        if which=='tf':
            
            test_tf = tfidf_vec.transform(df['test'])
            test_df = pd.DataFrame(test_tf.toarray())
            return test_df


# In[8]:



def _labeling(df):
    
    lis = df['labels'].values
    final = np.zeros(len(lis))
    
    for i in range(len(lis)):
        if(lis[i] == 1 or lis[i] == 2):
            final[i] = 1
        if(lis[i] == 3):
            final[i] = 2
        if(lis[i] == 4 or lis[i] == 5):
            final[i] = 3
    
    df['labels'] = pd.DataFrame(final).values
    return df


# In[59]:


def _callVecs(data_sample, which):
    
    if which == 'cv':
        Cvec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000).fit(data_sample['text'])
        filename = 'cv.pkl'
        pickle.dump(Cvec, open(filename, 'wb'))
        cv_df = _vectorizers('cv',data_sample,'train', Cvec)
        _knn(3, 1000, cv_df, 'cv_knn.pkl')
        
    elif which == 'tf':
        tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000).fit(data_sample['text'])
        filename = 'tfVec.pkl'
        pickle.dump(tfidf_vec, open(filename, 'wb'))
        tf_df = _vectorizers('tf',data_sample,'train', tfidf_vec)
        _knn(3, 1000, tf_df, 'tf_knn.pkl')
        


# In[60]:


def _knn(n_neigh, n_feat, dataframe, filename):

    knn = KNeighborsClassifier(n_neighbors=n_neigh, metric='euclidean').fit(dataframe.iloc[:,0:n_feat:1],dataframe['labels'])
    pickle.dump(knn, open(filename, 'wb'))
    


# In[62]:


def predict_knn(model, text, which):
    
    rev = pd.DataFrame(text, columns=['test'])
    if which == 'cv':
        dout = open('cv.pkl', 'rb')
        Cvec = pickle.load(dout)
        dout.close()
        vecs = _vectorizers(which, rev, 'test', Cvec )
    if which == 'tf':
        dout = open('tfVec.pkl', 'rb')
        tfidf_vec = pickle.load(dout)
        dout.close()
        vecs = _vectorizers(which, rev, 'test', tfidf_vec )
        
    return model.predict(vecs)


# In[64]:



def predict_gender(names):
    gender = []
    for i in names:
        gender.append(classify_gender(i.lower()))
    return gender
    


# In[229]:


def _graph_gen():
    graphs = pd.ExcelFile('review.xls')
    data = graphs.parse('reviews')
    
    gender_graph = dict(data['gender'].value_counts())
    ratings_graph = dict(data['rating'].value_counts())
    for i in range(1,4):
        if i not in ratings_graph.keys():
            ratings_graph[i] = 0
    gender_graph = [gender_graph['M'], gender_graph['F']]
    
    fig, ax = plt.subplots(figsize=(8, 8))

    g = ["Male","Female"]

    data = gender_graph
    ingredients = [x.split()[-1] for x in g]

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d} )".format(pct, absolute)
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    ax.legend(wedges, ingredients,
              title="Genders",
              loc = "center left",
              bbox_to_anchor=(0, 0.4, 0.5, 1))

    plt.setp(autotexts, size=15, weight="bold")

    ax.set_title("Gender Statistics")
    fig.savefig('users/static/users/pie.png')
    
    x = np.arange(3)
    money = [ratings_graph[1], ratings_graph[2], ratings_graph[3]]
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%d' % (x)
    formatter = FuncFormatter(millions)
    fig, ax = plt.subplots(figsize=(8, 6) )
    ax.yaxis.set_major_formatter(formatter)
    plt.bar(x, money, color = 'green')
    plt.xlabel('Class Label')
    plt.ylabel('Ratings Count')
    plt.title('Classification Counts')
    plt.xticks(x, ('-Ve(Ratings of 1 or 2)', 'Neutral(Ratings of 3)', '+ve(Ratings of 4 or 5)'))
    fig.savefig('users/static/users/bar.png')
    # matplotlib.pyplot.close(fig)
    plt.close(fig)
    # return ""
# In[230]:


def _functionTorun():

    file = 'review.xls'
    dout = open('cv_knn.pkl', 'rb')
    model = pickle.load(dout)
    dout.close()
    
    which = 'cv' # To indicate tfidf we are using tf as a flag, 
                 # use cv to indicate Count vectorizer
    
    # Load spreadsheet
    xl = pd.ExcelFile(file)
    # Load a sheet into a DataFrame by name: df1
    data = xl.parse('reviews')
    # type(data)
    data = data.values.tolist()
    g = []
    r = []
    i = []
    for record in data:
        if record[3]=='M' or record[3]=='F':
            continue
        else:
            i.append(record[0])
            g.append(record[1])
            r.append(record[2])
    print(i)
    if g:
        genders = predict_gender(g)
        ratings = list(predict_knn(model, r, 'cv'))
    
        count = 0
        wb = xlrd.open_workbook("review.xls")
        sheet = wb.sheet_by_index(0)
        final_data = []
        for idx in range(1,len(i)):
            print(idx, genders[count], ratings[count])
            row = sheet.row_values(idx)
            row[3] = genders[count]
            row[4] = ratings[count]
            count += 1
            row[0] = int(row[0])
            final_data.append(row)
#         for each in final_data:
#             print(each)

        rb = xlrd.open_workbook('review.xls')
        wb = copy(rb)
        s = wb.get_sheet(0)
        # print(final_data)
        i = 0
        for ind in final_data:
            i = i+1
            s.write(i,3,ind[-2])
            s.write(i,4,str(ind[-1]))
        wb.save('review.xls') 
    _graph_gen()
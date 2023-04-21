import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from time import localtime, strftime
from scipy.stats import spearmanr,pearsonr
import zipfile
import gc
import nltk
# fixing random seed for reproducibility
random.seed(123)
np.random.seed(123)


def read(path):

    f = open(path,'r',encoding='utf-8') #mac不用这行，直接pd.read_csv(path)
    data = pd.read_csv(f).iloc[:,1].values
    return data
def read_label(path):

    f = open(path,'r',encoding='utf-8') #mac不用这行，直接pd.read_csv(path)
    data = pd.read_csv(f).iloc[:,0].values #把pandas转成numpy操作 iloc
    return data


train_df = read(path = './data_topic/train.csv') #read value
test_df = read(path = './data_topic/test.csv')
dev_df = read('./data_topic/dev.csv')

train_label = read_label(path = './data_topic/train.csv') #read label
test_label = read_label(path = './data_topic/test.csv')
dev_label = read_label('./data_topic/dev.csv')
stop_words = ['a','in','on','at','and','or', 
              'to', 'the', 'of', 'an', 'by', 
              'as', 'is', 'was', 'were', 'been', 'be', 
            'are','for', 'this', 'that', 'these', 'those', 'you', 'i', 'if',
             'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
              'but', 'not', 'there', 'no', 'does', 'not', 'so', 've', 'their',
             'his', 'her', 'they', 'them', 'from', 'with', 'its']

def extract_ngrams(x_raw, ngram_range=(1,2), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', 
                   stop_words=[], vocab=set()):
    clf = []
    ngram_low = ngram_range[0]
    ngram_high = ngram_range[1]
    sent = re.findall(token_pattern, x_raw.lower()) # lowercase, remove punctuation & non-alphanumeric characters 
    for i, word in enumerate(sent):  # remove stoplist word 
        if word not in stop_words:
            clf.append(word)
    #print("the result is:{}".format（clf）)
    unigram = []
    bigrams = []
    x = [unigram,bigrams]
    
    for i,word in enumerate(clf):
        
        unigram.append(clf[i])
    # print('unigram:{}'.format(unigram))

    for i in range(len(clf)):
        
        bigrams.append(' '.join(clf[i:i+ngram_high]))
        i+=2
    return x
# result = extract_ngrams(x_raw = train_df, stop_words = stop_words)
# print(result)

def get_vocab(X_raw, ngram_range=(1,2), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', 
              min_df=0, keep_topN=10, 
              stop_words=[]):
    voc_uni = []
    voc_bi = []
    df = {}
    count_uni=0
    count_bi=0
    vocab = [voc_uni,voc_bi]
    for i,sent in enumerate(X_raw): #cycle to get each sentence 
        extract_ngram = extract_ngrams(sent,stop_words=stop_words)
        voc_uni.append(extract_ngram[0])#get unigram list
        voc_bi.append(extract_ngram[1]) #get bigrams list
        set_sent = set(sent)
    ngram_counts = {}
    for word in voc_uni:
        for words in word:
            if words in ngram_counts:
                ngram_counts[words]+=1
            else:
                ngram_counts[words]=0  
            for word_ in set_sent: #calculate document frequency 
                if words in df:
                    df[words]+=1
                else:
                    df[words]=1
                    
    for word in voc_bi:#use dictionary to collect the ngram and corresponding frequency  
        for words in word:
            if words in ngram_counts:
                ngram_counts[words]+=1
            else:
                ngram_counts[words]=0 
            for word_ in set_sent: #calculate document frequency 
                if words in df:
                    df[words]+=1
                else:
                    df[words]=1   
    for words in df: #set min_df to filter the value
        if df[words]<min_df:
            del df[words]
    if keep_topN!=0: #set keep_topN to filter the most N frequent words
        ngram_counts_top = sorted(ngram_counts.items(),key = lambda item:item[1],reverse=True)
        ngram_counts = []
        # print(ngram_counts_top[0])
        for i in range(keep_topN):
            ngram_counts.append(ngram_counts_top[i])
    return vocab, df, ngram_counts
# h,g,k= get_vocab(train_df)
# print(h,g,k)
corpus_list = np.concatenate((train_df,test_df,dev_df))
vocab_list,df,ngram_counts=get_vocab(corpus_list,stop_words = stop_words)
vocab_all = []
for words in vocab_list[0]:
    vocab_all.append(words)
for words in vocab_list[1]:
    vocab_all.append(words)
vocab_all = sum(vocab_all,[])
# print(vocab_all)

#doc_id and id_doc
vocab_id = {}
id_vocab = {}
for i in range(len(vocab_all)):
    vocab_id[vocab_all[i]] = i
for j in range(len(vocab_all)):
    id_vocab[j]= vocab_all[j]
# print(vocab_id)

def get_eachword(X,ngram_range=(1,2), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', 
                   stop_words=stop_words, vocab=set()):
    X_list = []
    for sent in X:
        each_word = extract_ngrams(sent, ngram_range=(1,2), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', 
                   stop_words=[], vocab=set())
        each_uni_word = each_word[0]
        for word in each_uni_word:
            X_list.append(word)
    return X_list
lX_train = get_eachword(train_df)
lX_dev = get_eachword(dev_df)
lX_test = get_eachword(test_df)
# print(lX_train)

#convert train,test,dev into list of indices in vocabulary
def X_vec(word_list):
    dict_wordlist = []
    for i,word in enumerate(word_list):
        if word in vocab_id.keys():
            # dict_wordlist[word]=vocab_id[word]
            dict_wordlist.append([word,vocab_id[word]])
    return dict_wordlist
dict_train = X_vec(lX_train)
print(dict_train)
# print(lX_train[-1])
#这个地方长度不dui###################################没改 但是结果是对的上的
# print(len(lX_test))   
# print(len(dict_train))

#put label y in array
Y_train =[]
Y_test=[]
Y_dev= []
for index in train_label:
    Y_train.append([index])
for index in dev_label:
    Y_dev.append([index])
for index in test_label:
    Y_test.append([index])
# print(Y_train)
# db_train = X_train +Y_train
# print(db_train)

def network_weights(vocab_size=1000, embedding_dim=300, 
                    hidden_dim=[], num_classes=3, init_val = 0.5):
    #creat m*n matrix
    # W = np.zeros((vocab_size,embedding_dim))
    hid_dim = hidden_dim[0]
    W_in=np.random.uniform(-0.1,0.1,(vocab_size,embedding_dim))
    W_hid = np.random.uniform(-0.1,0.1,(embedding_dim,hid_dim))
    W_out = np.random.uniform(-0.1,0.1,(hid_dim,num_classes))
    W = (W_in,W_hid,W_out)
    return W

def softmax(z):
    sig = 1/(1+np.exp(-z))
    
    return sig

def categorical_loss(y, y_preds):
    l = 0
    for i in range(len(y)):
        loss = y[i]* np.log(y_preds)
        l+=loss
        i+=1
    l = l/i
    return l

def relu(z):
    z_ = z.copy()
    a= np.maximum(0,z_)
    return a
    
def relu_derivative(z):
    dz = []
    for i in range(len(z)):
        if z[i]<=0:
            term = 0
        else:
            term = 1
        dz.appendd(term)
    return dz

def dropout_mask(size, dropout_rate):
    dropout_vec = np.ones(size)
    drop_num = int(size * dropout_rate)
    # print(drop_num)
    for i in range(drop_num-1):
        rand = random.randint(0,size-1)
        dropout_vec[rand] = 0
    return dropout_vec

#forward pass network
def forward_pass(x, W, dropout_rate=0.2):
    
    
    out_vals = {}
    
    h_vecs = []
    a_vecs = []
    dropout_vecs = []

    
   
    
    return out_vals




              






# for sent in train_df:
#     X_train.append([sent])
# for sent in dev_df:
#     X_dev.append([sent]) 
# for sent in test_df:
#     X_test.append([sent])

# Y_train =[]
# Y_test= []
# Y_dev= []
# for index in train_label:
#     Y_train.append([index])
# for index in dev_label:
#     Y_dev.append([index])
# for index in test_label:
#     Y_test.append([index])
# # print(Y_test[0])
# # print(len(X_test),len(Y_test))
# def creat_data(data,label):
#     i=0
#     db = [] #creat a database for sentence X and label Y
#     for i in range(len(data)):
#         tr = data[i]
#         tr.extend(label[i])
#         db.append(tr)
#         i+=1
#     return db
# # db_test = creat_data(X_test,Y_test)
# # print(db_test)
# db_train = creat_data(X_train,Y_train)
# print(db_train)


from algorithm import recommendation
from algorithm import similarity
from preprocess import read_data
import domain

# from lxml import etree
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import gensim
import _pickle as pickle
from bs4 import BeautifulSoup
import util
import time
import pandas as pd
import json
import re

import sys


w2v = gensim.models.Word2Vec.load('data/w2v_model_stemmed') # pre-trained word embedding
idf = pickle.load(open('data/idf',"rb")) # pre-trained idf value of all words in the w2v dictionary

df = pd.read_csv("data/BIKER_train.QApair.csv")
#df = df[df["prediction"]==1]
#df_querys = pd.read_csv("data/test_queries_multi_min_5_max_10_ir_10.csv")
df_querys = pd.read_csv("data/Biker_test_filtered.csv")
#df_querys = pd.read_csv("data/test_queries_min_5_max_10_ir_10.csv")

queries = df_querys["title"].to_list()

df=df[~(df["title"].isin(queries))]


title_list=df.title.to_list()
apis_list=df.answer.to_list()
question_list = []


for x in range(len(title_list)):
    answers=[]
    for ans in eval(apis_list[x]):
        # ans=".".join(ans.split(".")[2:])
        answers.append(domain.Answer(0,0,ans,0))
    q = domain.Question(x,title_list[x],"",0,0,0,answers=answers)
    question_list.append(q)

questions = recommendation.preprocess_all_questions(question_list, idf, w2v) # matrix transformation


javadoc = pickle.load(open('data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = pickle.load(open('data/parent', 'rb')) # parent is a dict(), which stores the ids of each query's duplicate questions



title_list = df_querys["title"].to_list()
apis_list = df_querys["answer"].to_list()

#title_list = df_querys["title"].to_list()
#apis_list = df_querys["answer"].to_list()
# querys = read_data.read_querys_from_file()
#querys = querys[:10]

query_titles=[]
for q in title_list:
    query_titles.append(re.sub(r'\W+', '', q[0].lower().replace(' ', '')))
# query_titles = [q[0].lower().replace(' ', '') for q in querys]
# questions = [q for q in questions if q.title not in query_titles]

question_list=[]
for q in questions:
    # print(q.title)
    if re.sub(r'\W+', '', q.title.lower().replace(' ', '')) not in query_titles:
        question_list.append(q)

questions = question_list

print ('loading data finished')

mrr = 0.0
map = 0.0
tot = 0
count=0
result_list = []

total_Bi_Encoder_precision=[0]*4
total_Bi_Encoder_recall=[0]*4

for idx in range(len(title_list)):

    #query = item[0].title

    query = title_list[idx]
    true_apis = set(eval(apis_list[idx]))
    query_words = WordPunctTokenizer().tokenize(query.lower())
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    top_questions_title = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
    # print(top_questions, questions)
    top_questions = {}
    recommended_api=[]
    for k,v in top_questions_title.items():
        if v[1]!=query:
            top_questions[k] = v[1]
        #print(k,v)
        # title
        title_str = df.iloc[k]["title"]
        answer_str = df.iloc[k]["answer"]
        #print(title_str)
        #print(v[0])
        if title_str == v[0]:
            #print(answer_str)
            recommended_api+=eval(answer_str)
    #recommended_api = recommendation.recommend_api_processed(query_matrix, query_idf_vector, top_questions, questions, javadoc, javadoc_dict_methods,-1)


    #recommended_api = recommendation.recommend_api_baseline(query_matrix,query_idf_vector,javadoc,-1)

    top_k= len(recommended_api)
    
    Bi_Encoder_hit_list=[0]*top_k
    Bi_Encoder_hit_recall_list=[0]*top_k

    pos = -1
    tmp_map = 0.0
    hits = 0.0
    temp_true_apis=list(true_apis)
    
    len_api=len(temp_true_apis)
    
    for i,api in enumerate(recommended_api):
    
        if api in true_apis:
            Bi_Encoder_hit_list[i]=1
            
        if api in true_apis and pos == -1:
            pos = i+1
            
        if api in temp_true_apis:
            hits += 1
            Bi_Encoder_hit_recall_list[i]=1
            
            tmp_map += hits/(i+1)
            temp_true_apis.remove(api)
            print(tmp_map,hits,i+1)
           
    tmp_map /= len(true_apis)
    print(tmp_map,hits,i+1)
    tmp_mrr = 0.0
    if pos!=-1:
        tmp_mrr = 1.0/pos
    #print(tmp_mrr,tmp_map, pos, true_apis)
    
    temp_precision=[0]*4
    temp_recall=[0]*4
    
    for idx, n in enumerate([1,3,5,10]):
    # print(Bi_Encoder_hit_recall_list)
        temp_precision[idx] = sum(Bi_Encoder_hit_list[:n])/n
        temp_recall[idx] = sum(Bi_Encoder_hit_recall_list[:n])/(len_api)
        
    total_Bi_Encoder_precision = [x + y for (x, y) in zip(total_Bi_Encoder_precision, temp_precision)]
    
    total_Bi_Encoder_recall = [x + y for (x, y) in zip(total_Bi_Encoder_recall, temp_recall)]
    
    print(total_Bi_Encoder_precision)
    print(total_Bi_Encoder_recall)
    
    map += tmp_map
    mrr += tmp_mrr

    #print (tmp_mrr,tmp_map,pos,query,true_apis,len(recommended_api))


    # for i, api in enumerate(recommended_api):
    #     if i==10:
    #         break
    #     print api,'rank',i
    #     recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)



    top_questions_list = {}
    for k,v in top_questions_title.items():
        top_questions_list[v[0]] = v[1] 
    #print(top_questions_list)
    count+=1
    result_list.append([pos,query,true_apis,top_questions_list,recommended_api])
    print(count)
    # print (tmp_mrr,tmp_map,pos,query,true_apis,top_questions,recommended_api)

    # for i, api in enumerate(recommended_api):
    #     if i==10:
    #         break
    #     print api,'rank',i
    #     recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)

df = pd.DataFrame(result_list,columns =['pos', 'title','answer','top_questions','recommended_api']) 
df.to_csv("result/big_biker_test_test.csv")

total_Bi_Encoder_precision = [x/len(title_list) for x in total_Bi_Encoder_precision]
total_Bi_Encoder_recall = [x/len(title_list) for x in total_Bi_Encoder_recall]

print (total_Bi_Encoder_precision)
print (total_Bi_Encoder_recall)

print (mrr/len(df),len(df))
print (map/len(df))


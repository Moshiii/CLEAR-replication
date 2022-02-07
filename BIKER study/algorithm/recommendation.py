from lxml import etree
from nltk.stem import SnowballStemmer
from algorithm import similarity
from nltk.tokenize import WordPunctTokenizer
import gensim
import _pickle as pickle
from bs4 import BeautifulSoup
import util
import time
import math
from preprocess import read_data

def preprocess_all_questions(questions,idf,w2v):
    processed_questions = list()
    for question in questions:
        title_words = WordPunctTokenizer().tokenize(question.title.lower())
        if title_words[-1] == '?':
            title_words = title_words[:-1]
        if len(title_words) <= 3:
            continue
        title_words = [SnowballStemmer('english').stem(word) for word in title_words]
        question.title_words = title_words
        question.matrix = similarity.init_doc_matrix(question.title_words,w2v)
        question.idf_vector = similarity.init_doc_idf_vector(question.title_words,idf)
        processed_questions.append(question)

    return processed_questions


def preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v):


    for api in javadoc:

        javadoc_dict_classes[api.class_name] = api.package_name+'.'+api.class_name

        description_words = [SnowballStemmer('english').stem(word) for word in api.class_description]
        api.class_description_matrix = similarity.init_doc_matrix(description_words,w2v)
        api.class_description_idf_vector = similarity.init_doc_idf_vector(description_words,idf)
        for api_method in api.methods_descriptions_stemmed:
            api.methods_matrix.append(similarity.init_doc_matrix(api_method,w2v))
            api.methods_idf_vector.append(similarity.init_doc_idf_vector(api_method,idf))
        for api_method in api.methods:
            javadoc_dict_methods[api.class_name+'.'+api_method] = api.package_name+'.'+api.class_name+'.'+api_method

def get_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,javadoc_dict_description,idf,w2v):


    for api in javadoc:

        javadoc_dict_classes[api.class_name] = api.package_name+'.'+api.class_name

        class_description_words = [SnowballStemmer('english').stem(word) for word in api.class_description]
        api.class_description_matrix = similarity.init_doc_matrix(class_description_words,w2v)
        api.class_description_idf_vector = similarity.init_doc_idf_vector(class_description_words,idf)
        for api_method in api.methods_descriptions_stemmed:
            api.methods_matrix.append(similarity.init_doc_matrix(api_method,w2v))
            api.methods_idf_vector.append(similarity.init_doc_idf_vector(api_method,idf))
        for idx, api_method in enumerate(api.methods):
            javadoc_dict_methods[api.class_name+'.'+api_method] = api.package_name+'.'+api.class_name+'.'+api_method
            javadoc_dict_description[api.class_name+'.'+api_method] = " ".join(api.methods_descriptions_stemmed[idx])

def get_topk_questions(origin_query,query_matrix,query_idf_vector,questions,topk,parent):

    # this function returns a dictionary of the top-k most relevant questions of the query
    # the key is question id, the value is the similarity between the question and the query

    query_id = '-1'
    for question in questions:
        if question.title == origin_query or question.title in origin_query or origin_query in question.title:  # the same question should not appear in the dataset
            query_id = question.id
            if query_id not in parent:
                parent[query_id] = query_id

    relevant_questions = list()
    for question in questions:

        if query_id in parent and question.id in parent and parent[query_id] == parent[question.id]: #duplicate questions
            continue

        valid = False
        for answer in question.answers:
            if int(answer.score)>=0:
                valid = True
        if not valid:
            continue

        sim = similarity.sim_doc_pair(query_matrix,question.matrix, query_idf_vector, question.idf_vector)
        relevant_questions.append((question.id, question.title, sim))

    list_relevant_questions = sorted(relevant_questions, key=lambda question: question[2], reverse=True)

    # get the ids of top-k most relevant questions
    top_questions = dict()
    for i, item in enumerate(list_relevant_questions):
        top_questions[item[0]] = [item[1],item[2]]
        if i+1 == topk:
            break

    return top_questions


def summarize_api_method(api_method, top_questions, questions, javadoc,javadoc_dict_methods):
    for api in javadoc:
        for i, method in enumerate(api.methods):
            if api.package_name + '.' + api.class_name + '.' + method == api_method:
                print ('>>>JavaDoc<<<')
                print (api.methods_descriptions_pure_text[i].replace('\n',' ').replace('  ',' ').split('.')[0]+'.')
                break

    titles = dict()
    code_snippets = dict()

    method_pure_name = api_method.split('.')[-1]

    for question in questions:
        if question.id not in top_questions:
            continue

        contains_api = False

        for answer in question.answers:

            soup = BeautifulSoup(answer.body, 'html.parser', from_encoding='utf-8')



            links = soup.find_all('a')
            for link in links:
                link = link['href']
                if 'docs.oracle.com/javase/' in link and '/api/' in link and 'html' in link:
                    pair = util.parse_api_link(link)  # pair[0] is class name, pair[1] is method name

                    if pair[1] != '':
                        method_name = pair[0] + '.' + pair[1]
                        if method_name == api_method:
                            titles[question.title] = top_questions[question.id]
                            contains_api = True

            codes = soup.find_all('code')
            for code in codes:
                code = code.get_text()
                pos = code.find('(')
                if pos != -1:
                    code = code[:pos]
                if code in javadoc_dict_methods:
                    method_name = javadoc_dict_methods[code]
                    if method_name == api_method:
                        titles[question.title] = top_questions[question.id]
                        contains_api = True

        if contains_api:
            snippet_list = list()
            for answer in question.answers:
                soup = BeautifulSoup(answer.body, 'html.parser', from_encoding='utf-8')
                code_snippet = soup.find('pre')
                if code_snippet is not None and code_snippet.get_text().count('\n') <= 5 \
                        and '.'+method_pure_name+'(' in code_snippet.get_text():
                    snippet_list.append(code_snippet.get_text())
            code_snippets[question.title] = snippet_list

    titles = sorted(titles.items(), key=lambda item: item[1], reverse=True)

    print ('>>>Relevant Questions<<<')
    tot = 0
    for i, title in enumerate(titles):
        if tot == 3:
            break
        if len(code_snippets[title[0]])>0:
            tot+=1
            print (str(tot)+'.'+title[0])

    if tot<3:
        for i, title in enumerate(titles):
            if tot == 3:
                break
            if len(code_snippets[title[0]])==0:
                tot += 1
                print (str(tot)+'.'+title[0])


    tot = 0
    for i, title in enumerate(titles):
        if tot == 3:
            break
        if len(code_snippets[title[0]]) > 0:
            tot += 1
            if tot == 1:
                print ('>>>Code Snippets<<<')
            print ('/**********code snippet', tot, '**********/')
            print (code_snippets[title[0]][0])

    if tot==0:
        print ('\n-----------------------------------------------\n')
    else: print ('-----------------------------------------------\n')




def recommend_api(query_matrix,query_idf_vector,top_questions,questions,javadoc,javadoc_dict_methods,topk):
    # remember that top_questions is a dictionary of the top-k most relevant questions of the query
    # the key is question id, the value is the similarity between the question and the query
    # questions is a list including all questions (api related) in StackOverflow
    # javadoc is a list including all api classes

    api_methods = dict() #stores the SO_sim of api method and the query
    api_methods_count = dict()

    for question in questions:
        if question.id not in top_questions:
            continue

        tmp_set = set()

        for answer in question.answers:

            if int(answer.score)<0:
                continue

            soup = BeautifulSoup(answer.body, 'html.parser', from_encoding='utf-8')
            links = soup.find_all('a')
            for link in links:
                link = link['href']
                if 'docs.oracle.com/javase/' in link and '/api/' in link and 'html' in link:
                    pair = util.parse_api_link(link)  # pair[0] is class name, pair[1] is method name

                    if pair[1] != '':
                        method_name = pair[0] + '.' + pair[1]
                        if method_name in tmp_set:
                            continue
                        else:
                            tmp_set.add(method_name)
                            if method_name in api_methods:
                                api_methods[method_name] += top_questions[question.id]
                                api_methods_count[method_name] += 1
                            else:
                                api_methods[method_name] = top_questions[question.id]
                                api_methods_count[method_name] = 1.0


            codes = soup.find_all('code')
            for code in codes:
                code = code.get_text()
                pos = code.find('(')
                if pos != -1:
                    code = code[:pos]

                if code in javadoc_dict_methods:
                    method_name = javadoc_dict_methods[code]
                    if method_name in tmp_set:
                        continue
                    else:
                        tmp_set.add(method_name)
                        if method_name in api_methods:
                            api_methods[method_name] += top_questions[question.id]
                            api_methods_count[method_name] += 1
                        else:
                            api_methods[method_name] = top_questions[question.id]
                            api_methods_count[method_name] = 1.0


    for key,value in api_methods.items():
    
        api_methods[key] = min(1.0, value/api_methods_count[key] * (1.0 + math.log(api_methods_count[key],2)/10))


    api_sim = {}

    for api in javadoc:
        class_name = api.package_name + '.' + api.class_name

        for i, method in enumerate(api.methods):

            method_name = class_name + '.' + method

            if method_name not in api_methods:
                continue
            else:
                doc_sim = similarity.sim_doc_pair(query_matrix,api.methods_matrix[i],query_idf_vector,api.methods_idf_vector[i])
                so_sim = api_methods[method_name]


                if method_name in api_sim:
                    api_sim[method_name] = max(api_sim[method_name],
                                                                 2 * doc_sim * so_sim / (doc_sim + so_sim))
                else:
                    api_sim[method_name] = 2 * doc_sim * so_sim / (doc_sim + so_sim)


    api_sim = sorted(api_sim.items(), key=lambda item: item[1], reverse=True)

    recommended_api = list()

    for item in api_sim:
        recommended_api.append(item[0])

        if topk!=-1 and len(recommended_api) >= topk:
            break

    return recommended_api

def recommend_api_processed(query_matrix,query_idf_vector,top_questions,questions,javadoc,javadoc_dict_methods,topk):
    # remember that top_questions is a dictionary of the top-k most relevant questions of the query
    # the key is question id, the value is the similarity between the question and the query
    # questions is a list including all questions (api related) in StackOverflow
    # javadoc is a list including all api classes

    api_methods = dict() #stores the SO_sim of api method and the query
    api_methods_count = dict()

    for question in questions:
        if question.id not in top_questions:
            continue

        tmp_set = set()

        for answer in question.answers:
            method_name = str(answer)

            tmp_set.add(method_name)
            if method_name in api_methods:
                api_methods[method_name] += top_questions[question.id]
                api_methods_count[method_name] += 1
            else:
                api_methods[method_name] = top_questions[question.id]
                api_methods_count[method_name] = 1.0
                
    for key,value in api_methods.items():
        #print(key,value)
        api_methods[key] = min(1.0, value/api_methods_count[key] * (1.0 + math.log(api_methods_count[key],2)/10))

    api_sim = {}

    for api in javadoc:
        class_name = api.package_name + '.' + api.class_name
        
        for i, method in enumerate(api.methods):

            method_name = class_name + '.' + method
            
            if method_name not in api_methods:
                continue
                
            else:
                doc_sim = similarity.sim_doc_pair(query_matrix,api.methods_matrix[i],query_idf_vector,api.methods_idf_vector[i])
                so_sim = api_methods[method_name]


                if method_name in api_sim:
                    api_sim[method_name] = max(api_sim[method_name],
                                                                 2 * doc_sim * so_sim / (doc_sim + so_sim))
                else:
                    api_sim[method_name] = 2 * doc_sim * so_sim / (doc_sim + so_sim)

    api_sim = sorted(api_sim.items(), key=lambda item: item[1], reverse=True)
    recommended_api = list()

    for item in api_sim:
        recommended_api.append(item[0])

        if topk!=-1 and len(recommended_api) >= topk:
            break

    return recommended_api





def recommend_api_class(query_matrix,query_idf_vector,top_questions,questions,javadoc,javadoc_dict_classes,topk):
    # remember that top_questions is a dictionary of the top-k most relevant questions of the query
    # the key is question id, the value is the similarity between the question and the query
    # questions is a list including all questions (api related) in StackOverflow
    # javadoc is a list including all api classes

    api_classes_count = dict()
    api_classes = dict() # stores the similarity between the question (whose answer contains the API class) and the query

    for question in questions:
        if question.id not in top_questions:
            continue
        for answer in question.answers:
            if int(answer.score)<0:
                continue

            soup = BeautifulSoup(answer.body, 'html.parser', from_encoding='utf-8')

            links = soup.find_all('a')
            for link in links:
                link = link['href']
                if 'docs.oracle.com/javase/' in link and '/api/' in link and 'html' in link:
                    pair = util.parse_api_link(link)  # pair[0] is class name, pair[1] is method name
                    class_name = pair[0] #note that this class_name already contains package name, i.e, java.util.Calendar
                    if class_name in api_classes:
                        api_classes[class_name] += top_questions[question.id]
                        api_classes_count[class_name] += 1
                    else:
                        api_classes[class_name] = top_questions[question.id]
                        api_classes_count[class_name] = 1

            codes = soup.find_all('code')
            for code in codes:
                code = code.get_text()
                pos = code.find('(')
                if pos != -1:
                    code = code[:pos]
                #code = code.replace('()', '')
                if code in javadoc_dict_classes:
                    # print code,'!class'
                    class_name = javadoc_dict_classes[code]
                    if class_name in api_classes:
                        api_classes[class_name] += top_questions[question.id]
                        api_classes_count[class_name] += 1
                    else:
                        api_classes[class_name] = top_questions[question.id]
                        api_classes_count[class_name] = 1

    for key,value in api_classes.items():
        api_classes[key] = min(1.0, value/api_classes_count[key] * (1.0 + math.log(api_classes_count[key],2)/10))

    api_sim = {}

    for api in javadoc:
        if api.package_name+'.'+api.class_name not in api_classes:
            continue

        doc_sim = 0.0

        for i,method_matrix in enumerate(api.methods_matrix):
            doc_sim = max(doc_sim,similarity.sim_doc_pair(query_matrix,method_matrix,query_idf_vector,api.methods_idf_vector[i]))

        so_sim = api_classes[api.package_name+'.'+api.class_name]


        api_sim[api.package_name+'.'+api.class_name] = 2 * doc_sim * so_sim / (doc_sim + so_sim)

        #the following code only considers SO similarity
        #api_sim[api.package_name + '.' + api.class_name] = so_sim



    api_sim = sorted(api_sim.items(), key=lambda item: item[1], reverse=True)

    recommended_api = list()

    for item in api_sim:
        recommended_api.append(item[0])
        if topk!=-1 and len(recommended_api) >= topk:
            break

    return recommended_api


def recommend_api_class_baseline(query_matrix,query_idf_vector,javadoc,topk):

    # class-level recommendation

    # this recommendation only considers javadoc similarity


    api_sim = {}

    for api in javadoc:

        doc_sim = 0.0

        for i, method_matrix in enumerate(api.methods_matrix):
            doc_sim = max(doc_sim, similarity.sim_doc_pair(query_matrix, method_matrix, query_idf_vector,
                                                           api.methods_idf_vector[i]))

        api_sim[api.package_name+'.'+api.class_name] = doc_sim

    api_sim = sorted(api_sim.items(), key=lambda item: item[1], reverse=True)

    recommended_api = list()

    for item in api_sim:
        recommended_api.append(item[0])
        if topk != -1 and len(recommended_api) >= topk:
            break

    return recommended_api


def recommend_api_baseline(query_matrix,query_idf_vector,javadoc,topk):

    # method-level recommendation

    # this recommendation only considers javadoc similarity

    api_sim = {}

    for api in javadoc:
        for i, method in enumerate(api.methods):
            doc_sim = similarity.sim_doc_pair(query_matrix,api.methods_matrix[i],query_idf_vector,api.methods_idf_vector[i])
            api_sim[api.package_name+ '.' +api.class_name + '.' + method] = doc_sim

    api_sim = sorted(api_sim.items(), key=lambda item: item[1], reverse=True)

    recommended_api = list()

    for item in api_sim:
        recommended_api.append(item[0])
        if topk!=-1 and len(recommended_api) >= topk:
            break

    return recommended_api








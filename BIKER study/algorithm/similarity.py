from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import gensim
import _pickle as pickle
import numpy as np
from sklearn import preprocessing


def init_doc_matrix(doc,w2v):

    matrix = np.zeros((len(doc),100)) #word embedding size is 100
    for i, word in enumerate(doc):
        if word in w2v.wv.vocab:
            matrix[i] = np.array(w2v.wv[word])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(doc)

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))

    return matrix

def init_doc_idf_vector(doc,idf):
    idf_vector = np.zeros((1,len(doc)))  # word embedding size is 100
    for i, word in enumerate(doc):
        if word in idf:
            idf_vector[0][i] = idf[word][1]

    return idf_vector



def sim_doc_pair(matrix1,matrix2,idf1,idf2):

    sim12 = (idf1*(matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()

    sim21 = (idf2*(matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()


    return 2 * sim12 * sim21 / (sim12 + sim21)
    total_len = matrix1.shape[0] + matrix2.shape[0]
    return sim12 * matrix2.shape[0] / total_len + sim21 * matrix1.shape[0] / total_len



if __name__ == "__main__":
    w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed')

    idf = pickle.load(open('../data/idf'))



    question1 = 'intialize all elements in an ArrayList as a specific integer'
    question1 = WordPunctTokenizer().tokenize(question1.lower())
    question1 = [SnowballStemmer('english').stem(word) for word in question1]

    question2 = 'set every element of a list to the same constant value'
    question2 = WordPunctTokenizer().tokenize(question2.lower())
    question2 = [SnowballStemmer('english').stem(word) for word in question2]

    matrix1 = init_doc_matrix(question1,w2v)
    matrix2 = init_doc_matrix(question2,w2v)
    matrix1_trans = matrix1.T
    matrix2_trans = matrix2.T

    idf1 = init_doc_idf_vector(question1,idf)
    idf2 = init_doc_idf_vector(question2,idf)

    #print sim_question_api(question1, question2, idf, w2v)
    print (sim_doc_pair(matrix1, matrix2, idf1, idf2))

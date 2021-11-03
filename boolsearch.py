from PorterStemmer import *
import os
import snappy
import re
import json
from bs4 import BeautifulSoup as bs
from collections import OrderedDict
from invidx_cons import *

# def read_compression_idx(filepath):

#     f = open(filepath,"rb")
#     data = f.read()

#     return data.decode()

def read_compression_3idx(filepath):

    f = open(filepath,"rb")
    data = f.read()

    return data

def read_dict(filepath):

    with open(filepath, "r") as jsonfile:
        temp = json.load(jsonfile)
    # print(temp)
    return temp

def preprocess(word):

    p = PorterStemmer()
    word = word.lower()
    word = p.stem(word,0,len(word)-1)

    return word

def single_word_retrieval(word,inv_dict):
    if word in inv_dict:
        ans = inv_dict[word]
    else:
        ans = []
    return ans

def multi_word_retrieval(word,p2,inv_dict):
    
    p1 = single_word_retrieval(word,inv_dict)
    i = 0
    j = 0
    res = []
    while (i<len(p1) and j<len(p2)):
        if (p1[i]==p2[j]):
            res.append(p1[i])
            i+=1
            j+=1
        elif (p1[i]>p2[j]):
            j+=1
        else:
            i+=1

    return res

def multi_keyword(words,inv_dict):
    
    res = single_word_retrieval(words[0],inv_dict)
    if (len(res)==0):
        return []
    for i in range (1,len(words)):
        res = multi_word_retrieval(words[i],res,inv_dict)
        if (len(res)==0):
            return [] 
    return res

def process_c0(query,data_dict,data,doc_map,stopwords,skip):
    new_dict = {}
    words = re.split(r'''[ ',.:();{}'`"\n]''',query)
    all_words = []
    

    for w in words:
        if (len(w)>0 and stopwords.get(w) is None):
            temp = preprocess(w)
            all_words.append(temp)

    for word in all_words:

        res = single_word_retrieval(word,data_dict)
        if (len(res)==0):
            return []

        ans = c0_decode(str(data[res[0]:res[0]+res[1]]))

        new_dict[word] = ans

    res = multi_keyword(all_words,new_dict)

    for i in range (0,len(res)):
        res[i] = doc_map[str(res[i])]

    return res

def process_c1(query,data_dict,data,doc_map,stopwords,skip):
    new_dict = {}
    words = re.split(r'''[ ',.:();{}'`"\n]''',query)
    all_words = []
    # data = data[skip:]
    # data = data.zfill(add+len(data))

    for w in words:
        if (len(w)>0 and stopwords.get(w) is None):
            temp = preprocess(w)
            all_words.append(temp)

    for word in all_words:

        res = single_word_retrieval(word,data_dict)
        if (len(res)==0):
            return []

        ans = c1_decode(str(data[res[0]:res[0]+res[1]]))
        ans = rev_gap_single(ans)


        new_dict[word] = ans

    res = multi_keyword(all_words,new_dict)

    for i in range (0,len(res)):
        res[i] = doc_map[str(res[i])]

    return res

def process_c2(query,data_dict,data,doc_map,stopwords,skip):
    new_dict = {}
    words = re.split(r'''[ ',.:();{}'`"\n]''',query)
    all_words = []
    

    for w in words:
        if (len(w)>0 and stopwords.get(w) is None):
            temp = preprocess(w)
            all_words.append(temp)

    for word in all_words:

        res = single_word_retrieval(word,data_dict)
        
        if (len(res)==0):
            return []

        ans = c2_decode(str(data[res[0]:res[0]+res[1]]))
        ans = rev_gap_single(ans)

        new_dict[word] = ans

    res = multi_keyword(all_words,new_dict)

    for i in range (0,len(res)):
        res[i] = doc_map[str(res[i])]

    return res

def process_c3(query,data_dict,data,doc_map,stopwords):
    new_dict = {}
    words = re.split(r'''[ ',.:();{}'`"\n]''',query)
    all_words = []

    for w in words:
        if (len(w)>0 and stopwords.get(w) is None):
            temp = preprocess(w)
            all_words.append(temp)

    for word in all_words:

        res = single_word_retrieval(word,data_dict)
        if (len(res)==0):
            return []

        ans = data[res[0]:res[0]+res[1]]
        ans = rev_gap_single(ans)
        new_dict[word] = ans

    res = multi_keyword(all_words,new_dict)

    for i in range (0,len(res)):
        res[i] = doc_map[str(res[i])]

    return res

def conv_num_to_binary(num):

    return "{0:b}".format(num)

if __name__ == "__main__":

    dict_1 = read_dict(sys.argv[4])
    data_dict = dict_1["inv"]
    doc_map = dict_1["map"]
    stopword = dict_1["stopwords"]
    comp = dict_1["compression"]


    if (comp!=3):
        skip = dict_1["to_skip"]
        path = sys.argv[3]
        data_list = []
        with open(path,"rb") as f:
            
            while True:
                byte = f.read(1)
                if not byte:
                    break
                val = int.from_bytes(byte,'big')
                res = conv_num_to_binary(val)
                res = res.zfill(8)
                data_list.append(res)

        data = str(''.join(data_list))
        temp = len(data)
        data = data[0:temp-skip]

    else:
        data = read_compression_3idx(sys.argv[3])

    f = open(sys.argv[1],"r")
    lines = f.readlines()
    queries = []

    o = open(sys.argv[2],"w")

    for line in lines:
        queries.append(line.strip())
    i = 0

    if (comp ==3):
        data_res = c3_decode(data)

    for query in queries:
        
        if (comp == 1):
            ans = process_c1(query,data_dict,data,doc_map,stopword,skip)

        elif (comp == 2):
            ans = process_c2(query,data_dict,data,doc_map,stopword,skip)

        elif (comp == 0):
            ans = process_c0(query,data_dict,data,doc_map,stopword,skip)

        elif (comp == 3):
            ans = process_c3(query,data_dict,data_res,doc_map,stopword)

        for a in ans:
            o.write("Q"+str(i)+" ")
            o.write(a +" "+ "1.0")
            o.write("\n")
        i+=1
    
import os
import re
import sys
from bs4 import BeautifulSoup as bs
import json
import csv
import time
import nltk
from nltk.corpus import stopwords
import random
import numpy as np
import math
from collections import OrderedDict
from numpy import linalg as LNG 
from nltk.corpus import stopwords

# get_text from a file
def get_text(filepath,docID):
    
    path = filepath
    meta_path = os.path.join(path,"metadata.csv")
    
    with open(meta_path,'r',encoding='utf-8') as f_in:
        text = []
        reader = csv.DictReader(f_in)
        for row in reader:
            cord_uid = row['cord_uid']
            
            if (cord_uid == docID):
                if row['pmc_json_files']:
                    for json_path in row['pmc_json_files'].split('; '):
                        path_split = list(json_path.split('/'))
                        json_path = path
                        for p in path_split:
                            json_path = os.path.join(json_path,p)
                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)
                            for paragraph in full_text_dict['body_text']:
                                paragraph_text = paragraph['text']
                                text.append(paragraph_text)
                    
                elif row['pdf_json_files']:
                    for json_path in row['pdf_json_files'].split('; '):
                        path_split = list(json_path.split('/'))
                        json_path = path
                        for p in path_split:
                            json_path = os.path.join(json_path,p)
                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)
                            for paragraph in full_text_dict['body_text']:
                                paragraph_text = paragraph['text']
                                text.append(paragraph_text)
                    
                elif row['abstract']:
                    text.append(row['abstract'])
                    
                elif row['title']:
                    
                    text.append(row['title'])
                break

        return text

# Get Text from given doc path from cordid to pmc dictionary
def get_text_2(filepath,docID,cordid_to_pmc):
    
    text = []
    if (isinstance(cordid_to_pmc[docID],list)):
        for json_path in cordid_to_pmc[docID]:
            with open(json_path) as f_json:
                full_text_dict = json.load(f_json)
                for paragraph in full_text_dict['body_text']:
                    paragraph_text = paragraph['text']
                    text.append(paragraph_text)
                    
    else:
        text.append(cordid_to_pmc[docID])
    
    return text

# get stopwords
def get_stopwords():
    stop_words = set(stopwords.words('english'))
    stop_words.add('&')
    stop_words.add('=')
    stop_words.add('"')
    stop_words.add('<')
    stop_words.add('>')

    return stop_words

# get df and corduid dictionary
def get_doc_freq(filepath):
    
    path = filepath
    meta_path = os.path.join(path,"metadata.csv")
    # pbar = tqdm(total=192509, position=0, leave=True)
    doc_freq = {}
    text = {}
    cordid_to_pmc = {}

    with open(meta_path,'r',encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)        
        for row in reader:
            # pbar.update(1)
            cord_uid = row['cord_uid']
            
            if cord_uid not in text:
                text[cord_uid] = 1
                text_doc = []
                if row['pmc_json_files']:
                    for json_path in row['pmc_json_files'].split('; '):
                        path_split = list(json_path.split('/'))
                        json_path = path
                        for p in path_split:
                            json_path = os.path.join(json_path,p)
                            
                        if cord_uid not in cordid_to_pmc:
                            cordid_to_pmc[cord_uid] = [json_path]
                        else:
                            cordid_to_pmc[cord_uid].append(json_path)
                            
                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)
                            for paragraph in full_text_dict['body_text']:
                                paragraph_text = paragraph['text']
                                text_doc.append(paragraph_text)

                elif row['pdf_json_files']:
                    for json_path in row['pdf_json_files'].split('; '):
                        path_split = list(json_path.split('/'))
                        json_path = path
                        for p in path_split:
                            json_path = os.path.join(json_path,p)
                            
                        if cord_uid not in cordid_to_pmc:
                            cordid_to_pmc[cord_uid] = [json_path]
                        else:
                            cordid_to_pmc[cord_uid].append(json_path)
                            
                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)
                            for paragraph in full_text_dict['body_text']:
                                paragraph_text = paragraph['text']
                                text_doc.append(paragraph_text)

                elif row['abstract']:

                    cordid_to_pmc[cord_uid] = row['abstract']
                    text_doc.append(row['abstract'])

                elif row['title']:
                    cordid_to_pmc[cord_uid] = row['title']
                    text_doc.append(row['title'])
                
                
                tokenised_terms = get_unique_terms(text_doc,stop_words)
                
                for token in tokenised_terms:
                    if token not in doc_freq:
                        doc_freq[token] = 1
                    else:
                        doc_freq[token]+=1
            
        # pbar.close()
        return doc_freq,cordid_to_pmc

def get_term_frequency(filepath,docid,stop_words,cordid_to_pmc):
    
    text_content = get_text_2(filepath,docid,cordid_to_pmc)
    term_freq = {}
    for text in text_content:
        tokenized_content = re.split(r'''[ ',-.:();{}?`"\n]''',text)
        for token in tokenized_content:
            token = token.lower()
            if (len(token)>2):
                if (token not in stop_words and not re.search('[0-9]+',token)):

                    if token not in term_freq:
                        term_freq[token] = 1
                    else:
                        term_freq[token] += 1
                    
    return term_freq

def get_unique_terms(text_content,stop_words):
    ans = set()
    for text in text_content:
        tokenized_content = re.split(r'''[ ',.-:();{}?`"\n]''',text)
        for token in tokenized_content:
            token = token.lower()
            if (len(token)>2):
                if (token not in stop_words and not re.search('[0-9]+',token)):
                        ans.add(token)
    unique_ans = list(ans)
                    
    return unique_ans

def get_total_docs(filepath):
    path = filepath
    meta_path = os.path.join(path,"metadata.csv")
    with open(meta_path,'r',encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        
        total_docs = len(list(reader))
    
    return total_docs

def get_tf(tf):
    
    normalised_tf = 1+math.log(tf,2)
    return normalised_tf

def get_idf(doc_freq_dict,term,total_doc):

    temp = 1 + (total_doc)/(doc_freq_dict[term])
    normalised_idf = math.log(temp,2)
    
    return normalised_idf

def get_doc_vector(filepath,docid,doc_freq_dict,total_doc,cordid_to_pmc):
    
    arr = get_term_frequency(filepath,docid,stop_words,cordid_to_pmc)
    vector = []

    for word in doc_freq_dict:
        if word in arr:
            vector.append(get_tf(arr[word])*get_idf(doc_freq_dict,word,total_doc))
        else:
            vector.append(0)
            
    return np.array(vector)

def get_sim(doc_vec,query_vec):
    
    dot_prod = np.dot(doc_vec,query_vec)
    norm_doc_vec = LNG.norm(doc_vec)
    norm_query_vec = LNG.norm(query_vec)
    
    sim = dot_prod/((norm_doc_vec)*(norm_query_vec))
    
    return abs(sim)

def get_norm(doc_vec):
    norm_doc_vec = LNG.norm(doc_vec)
    return norm_doc_vec

def get_normalised_doc_vector(filepath,docid,doc_freq_dict,total_doc,cordid_to_pmc):
    
    doc_vec = get_doc_vector(filepath,docid,doc_freq_dict,total_doc,cordid_to_pmc)
    norm_val = get_norm(doc_vec)
    if (norm_val!=0):
        norm_doc_vec = doc_vec/norm_val
    else:
        norm_doc_vec = 0
        
    return norm_doc_vec

def get_term_frequency_query(query,stop_words):

    text = query
    term_freq = {}

    tokenized_content = tokenized_content = re.split(r'''[ ',.-:();{}?`"\n]''',text)
    for token in tokenized_content:
        token = token.lower()
        if (len(token)>2):
            if (token not in stop_words and not re.search('[0-9]+',token)):
                if token not in term_freq:
                    term_freq[token] = 1
                else:
                    term_freq[token] += 1
                    
    return term_freq

def get_query_vector(query,vocab,doc_freq_dict,total_doc):
    
    arr = get_term_frequency_query(query,stop_words)
    vector = np.zeros(len(vocab))
    i = 0

    for word in arr:
        if word in doc_freq_dict:
            pos = vocab[word]
            vector[pos] = get_tf(arr[word])*get_idf(doc_freq_dict,word,total_doc)
            
    return vector

def get_normalised_query_vector(query,doc_freq_dict,total_doc):
    
    doc_vec = get_query_vector(query,doc_freq_dict,total_doc)
    norm_val = get_norm(doc_vec)
    if (norm_val!=0):
        norm_doc_vec = doc_vec/norm_val
    else:
        norm_doc_vec = 0
        
    return norm_doc_vec

def parse_query_doc(queryfile,field):
    query_path = queryfile
    all_queries = {}
    with open(query_path,"r") as f:

        content = f.read()
        bs_content = bs(content,'lxml')
        all_doc = bs_content.find_all('topic',{'number':True})
        for doc in all_doc:
            query_no = doc.attrs['number']
            query = doc.find(field).get_text()
            all_queries[query_no] = query
    return all_queries

def processtop100(top100):
    
    # curr = os.getcwd()
    path_txt = top100
    file_txt = open(path_txt,"r")
    query_relevant = {}
    while True:
        line = file_txt.readline()
        all_words = line.split()
        if not line or (len(all_words)==0):
            break

        if all_words[0] not in query_relevant:
            query_relevant[all_words[0]] = [all_words[2]]
        else:
            query_relevant[all_words[0]].append(all_words[2])
            
    return query_relevant

def get_all_doc_tf(filepath,query_relevant,cordid_to_pmc):
    
    train_doc_tf = {}
    for query in query_relevant:
        all_docs = query_relevant[query]
        # loop = tqdm(all_docs, position=0, leave=True)
        for doc in all_docs:
            if doc not in train_doc_tf:
                arr = get_term_frequency(filepath,doc,stop_words,cordid_to_pmc)
                train_doc_tf[doc] = arr
        
    return train_doc_tf

def get_vocab_vector(doc_freq_dict):
    i = 0
    vocab = {}
    # loop = tqdm(doc_freq_dict)
    for word in doc_freq_dict:
        vocab[word] = i
        i+=1
    return vocab

def get_doc_vector_train(filepath,docid,vocab,doc_freq_dict,total_doc,train_doc_tf):
    
    arr = train_doc_tf[docid]
    vector = np.zeros(len(doc_freq_dict))

    for word in arr:
        if word in vocab:
            pos = vocab[word]
            vector[pos] = get_tf(arr[word])*get_idf(doc_freq_dict,word,total_doc)
            
    return vector

def relevance_check(qrels,query_relevant):
    
    curr = os.getcwd()
    path_txt = os.path.join(curr,qrels)
    file_txt = open(path_txt,"r")
    rel_check = {}
    count = 0
    while True:
        
        line = file_txt.readline()
        all_words = line.split()
        if not line or (len(all_words)==0):
            break
        
        if all_words[2] in query_relevant[str(all_words[0])]:
            if all_words[2] in rel_check:
                rel_check[all_words[2]].append((all_words[0],all_words[3]))
            else:
                rel_check[all_words[2]] = [(all_words[0],all_words[3])]
        
    return rel_check
    
def get_dict_rels(query_relevant,queryno):
    
    dict_rels = {}
    for doc in query_relevant[queryno]:
        rel_val = 0
        if (doc in rel_check):
            check_rel = rel_check[doc]
            for t in check_rel:
                if (t[0]==queryno):
                    rel_val = t[1]
        dict_rels[doc] = rel_val
        
    return dict_rels

def get_doc_vector_sum(filepath,queryno,query_relevant,dict_rels):
    
    d_r = 0
    d_n = 0
    r = 0
    nr = 0
    # loop = tqdm(query_relevant[queryno], position=0, leave=True)
    doc_index = {}
    
    for doc in query_relevant[queryno]:
        
        d_i = get_doc_vector_train(filepath,doc,doc_freq_dict,total_doc,train_doc_tf)
        doc_index[doc] = d_i
        rel_val = dict_rels[doc]
        end = time.time()

        if int(rel_val)>0:
            d_r = np.add(d_r,d_i)
            r+=1
        else:
            d_n = np.add(d_n,d_i)
            nr+=1

    if (r!=0):
        d_r = d_r/r
    else:
        d_r = 0
    
    if (nr!=0):
        d_n = d_n/nr
    else:
        d_n = 0
        
    
    return d_r,d_n,doc_index

def formulated_query(d_r,d_n,q0,a,b,c):
    
    qr = a*q0 + b*d_r - c*d_n
    
    return qr

def calc_new_ranks(filepath,formulated_query,query_relevant,queryno,doc_freq_dict,total_doc,doc_index):
    
    new_ranks = {}
    all_ranks = []
    top_docs = query_relevant[queryno]
    # loop = tqdm(top_docs, position=0, leave=True)
    for doc in top_docs:
        rank_val = get_sim(formulated_query,doc_index[doc])
        if rank_val not in new_ranks:
            new_ranks[rank_val] = [doc]
        else:
            new_ranks[rank_val].append(doc)  
            
        all_ranks.append(rank_val)
    return all_ranks,new_ranks

def calc_ranklist(new_rank_list,new_rank_dict):
    
    new_rank_list = sorted(new_rank_list,reverse=True)
    output_rank = OrderedDict()
    for rank in new_rank_list:
        for rank_doc in new_rank_dict[rank]:
            output_rank[rank_doc] = rank
        
    return output_rank

def ndcg_val(rel,i):
    
    ans = int(rel)/math.log(i+1,2)
    
    return ans

def get_dict_rels(query_relevant,queryno):
    
    dict_rels = {}
    for doc in query_relevant[queryno]:
        rel_val = 0
        if (doc in rel_check):
            check_rel = rel_check[doc]
            for t in check_rel:
                if (t[0]==queryno):
                    rel_val = t[1]
        dict_rels[doc] = rel_val
        
    return dict_rels

def idcg(rel_check,queryno,query_relevant,dict_rels):

    ans = 0
    arr = []
    for doc in query_relevant[queryno]:
        if dict_rels[doc]=="1":
            arr.append(1)
        elif dict_rels[doc] == "2":
            arr.append(2)

    arr = sorted(arr,reverse=True)

    for i in range (0,len(arr)):
        ans += ndcg_val(arr[i],i+1)
    return ans

def calc_ndcg(ranklist,rel_check,queryno,dictrels,query_relevant):

    ans = 0
    i = 1
    for doc in ranklist:
        rel_val = dictrels[doc]
        val = ndcg_val(rel_val,i)
        i+=1
        ans+=val
    ideal = idcg(rel_check,queryno,query_relevant,dictrels)
    
    if (ideal==0):
        return 0
    return (ans/ideal)

def get_a_array(temp):
    
    arr = []

    start = math.floor(temp*5)
    end = math.ceil(temp*5)
    for i in range (-start,end):
        arr.append(round(temp+0.1*i,2))

    return arr

def get_b_array(k):
    
    arr = []

    start = math.floor(k*5)
    end = math.ceil(k*5)
    for i in range (-start,end):
        arr.append(round(k+i*0.1,2))
        
    return arr

def grid_search(filename,rel_check,queryno,dictrels,query_relevant,d_r,d_n,q0,doc_freq_dict,total_doc,doc_index,k1,k2,c):
    
    a_arr = get_a_array(k1)
    b_arr = get_b_array(k2)
    a_found = k1
    b_found = k2
    c = 0
    ndcg_val = 0
    
    for a in a_arr:
        for b in b_arr:
            q_new = formulated_query(d_r,d_n,q0,a,b,c)
            new_rank_list,new_rank_dict = calc_new_ranks(filename,q_new,query_relevant,queryno,doc_freq_dict,total_doc,doc_index)
            ranklist = calc_ranklist(new_rank_list,new_rank_dict)
            val = calc_ndcg(ranklist,rel_check,queryno,dictrels,query_relevant)
            
            if (val>ndcg_val):
                ndcg_val = val
                a_found = a
                b_found = b
                print(ndcg_val)
                
    return a_found,b_found

def precompute_centroid(filepath,train_doc_tf,vocab,doc_freq_dict,total_doc):
    d_ans = 0
    
    r = [random.choice(list(train_doc_tf)) for i in range(1000)]
    # loop = tqdm(r)
    for t in r:
        d_i = get_doc_vector_train(filepath,t,vocab,doc_freq_dict,total_doc,train_doc_tf)
        d_ans = np.add(d_ans,d_i)
        
    d_ans = d_ans/1000
    
    return d_ans

def calc_a_b(filepath,rel_check,query_relevant,doc_freq_dict,total_doc,total_epochs):
    
    a = 1
    b = 0.75
    for epoch in range (0,total_epochs):
        print(epoch)
        for q in query_relevant:
            print(q)
            dict_rels = get_dict_rels(query_relevant,q)
            d_r,d_n,doc_index = get_doc_vector_sum(filepath,q,query_relevant,dict_rels)
            q0 = get_query_vector(all_queries[q],vocab,doc_freq_dict,total_doc)
            a,b = grid_search(filepath,rel_check,q,dict_rels,query_relevant,d_r,d_n,q0,doc_freq_dict,total_doc,doc_index,a,b)
            print(a,b)
    
    return a,b

def get_rel_doc_sum_test(filepath,queryno,query_relevant,doc_freq_dict,total_doc,train_doc_tf):
    
    d_r = 0
    r = 0
    # loop = tqdm(query_relevant[queryno], position=0, leave=True)
    doc_index = {}
    
    for doc in query_relevant[queryno]:
        d_i = get_doc_vector_train(filepath,doc,vocab,doc_freq_dict,total_doc,train_doc_tf)
        doc_index[doc] = d_i
    
        d_r = np.add(d_r,d_i)
        r+=1

    if (r!=0):
        d_r = d_r/r
    else:
        d_r = 0
        
    return d_r,doc_index

def test_on_query(filepath,queryno,query_relevant,doc_freq_dict,total_doc,train_doc_tf,centroid_nonrelevant,all_queries,a,b,c):


    d_r,doc_index = get_rel_doc_sum_test(filepath,queryno,query_relevant,doc_freq_dict,total_doc,train_doc_tf)

    d_n = centroid_nonrelevant
    q0 = get_query_vector(all_queries[queryno],vocab,doc_freq_dict,total_doc)
    q_n = formulated_query(d_r,d_n,q0,a,b,c)

    all_ranks,new_ranks = calc_new_ranks(filepath,q_n,query_relevant,queryno,doc_freq_dict,total_doc,doc_index)
    ranklist = calc_ranklist(all_ranks,new_ranks)

    return ranklist

def calc_current_score(query_relevant,queryno,rel_check):

    ranklist = query_relevant[queryno]
    dict_rels = get_dict_rels(query_relevant,queryno)
    score = calc_ndcg(ranklist,rel_check,queryno,dict_rels,query_relevant)
    
    return score

def eval_score(ranklist,rel_check,queryno,query_relevant):

    dict_rels = get_dict_rels(query_relevant,queryno)
    score = calc_ndcg(ranklist,rel_check,queryno,dict_rels,query_relevant)

    return score

if __name__ == '__main__':

    stop_words = get_stopwords()
    queries = sys.argv[1]
    top100_file = sys.argv[2]
    # 2020-07-16 in dev
    filepath = sys.argv[3]

    doc_freq,cordid_to_pmc = get_doc_freq(filepath)
    doc_freq_dict = OrderedDict(doc_freq)
    total_doc = get_total_docs(filepath)
    all_queries = parse_query_doc(queries,"question")
    query_relevant = processtop100(top100_file)
    vocab = get_vocab_vector(doc_freq_dict)

    # print("calculating term frequencies")
    train_doc_tf = get_all_doc_tf(filepath,query_relevant,cordid_to_pmc)
    # print("calculating centroid")
    centroid_nonrelevant = precompute_centroid(filepath,train_doc_tf,vocab,doc_freq_dict,total_doc)
    # rel_check = relevance_check("t40-qrels.txt",query_relevant)

    a = 0.8
    b = 0.2
    c = 0.2
    o = open(sys.argv[4],"w")

    for query in query_relevant:
        i = 1
        # print(query)
        ranklist = test_on_query(filepath,query,query_relevant,doc_freq_dict,total_doc,train_doc_tf,centroid_nonrelevant,all_queries,a,b,c)
        for rank in ranklist:
            o.write(query+" ")
            o.write("Q0 ")
            o.write(str(rank) + " ")
            o.write(str(i) + " ")
            o.write(str(ranklist[rank]) + " ")
            o.write("runid1")
            o.write("\n")
            # print(rank + " ")
            # print(ranklist[rank])
            i+=1
        
        # print(ranklist)


        # print("New Score: ")
        # print(eval_score(ranklist,rel_check,query,query_relevant))
        # print("Old Score: ")
        # print(calc_current_score(query_relevant,query,rel_check))

    # ranklist = test_on_query("2020-07-16","2",query_relevant,doc_freq_dict,total_doc,train_doc_tf,centroid_nonrelevant,all_queries,a,b)
    # print("New Score: ")
    # print(eval_score(ranklist,rel_check,"2",query_relevant))
    # print("Old Score: ")
    # print(calc_current_score(query_relevant,"2",ranklist,rel_check))


    # ranklist = test_on_query("2020-07-16","3",query_relevant,doc_freq_dict,total_doc,train_doc_tf,centroid_nonrelevant,all_queries,a,b)
    # print("New Score: ")
    # print(eval_score(ranklist,rel_check,"3",query_relevant))
    # print("Old Score: ")
    # print(calc_current_score(query_relevant,"3",ranklist,rel_check))

    # ranklist = test_on_query("2020-07-16","4",query_relevant,doc_freq_dict,total_doc,train_doc_tf,centroid_nonrelevant,all_queries,a,b)
    # print("New Score: ")
    # print(eval_score(ranklist,rel_check,"4",query_relevant))
    # print("Old Score: ")
    # print(calc_current_score(query_relevant,"4",ranklist,rel_check))
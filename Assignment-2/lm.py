import os
import re
import sys
from bs4 import BeautifulSoup as bs
import json
import csv
import time
import nltk
from nltk.corpus import stopwords
# from tqdm import tqdm
import numpy as np
import math
from collections import OrderedDict
from numpy import linalg as LNG 
from nltk.corpus import stopwords

def get_doc_freq(filepath):
    
    path = filepath
    meta_path = os.path.join(path,"metadata.csv")

    doc_freq = {}
    text = {}
    cordid_to_pmc = {}

    with open(meta_path,'r',encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)        
        for row in reader:
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


                elif row['abstract']:
                    cordid_to_pmc[cord_uid] = row['abstract']


                elif row['title']:
                    cordid_to_pmc[cord_uid] = row['title']

        return cordid_to_pmc

def get_stopwords():
    stop_words = set(stopwords.words('english'))
    stop_words.add('&')
    stop_words.add('=')
    stop_words.add('"')
    stop_words.add('<')
    stop_words.add('>')

    return stop_words

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

def get_term_frequency_query(query,stop_words):

    text = query
    term_freq = {}

    tokenized_content = re.split(r'''[ ',.-:();{}?`"\n]''',text)
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

def get_all_doc_size(filepath,cordid_to_pmc,docs,stop_words):

    doc_size = {}
    for doc in docs:
        count = 0
        freq_in_doc = get_term_frequency(filepath,doc,stop_words,cordid_to_pmc)
        for term in freq_in_doc:
            count += freq_in_doc[term]
        
        doc_size[doc] = count

    return doc_size

def get_avg_doc_size(doc_size):
    
    avg_size = 0
    for doc in doc_size:
        
        avg_size += doc_size[doc]
        
    avg_size = avg_size/100
    
    return avg_size

def get_doc_tf(filepath,query_relevant,cordid_to_pmc,query):
    
    train_doc_tf = {}

    all_docs = query_relevant[query]
    # loop = tqdm(all_docs, position=0, leave=True)
    for doc in all_docs:
        if doc not in train_doc_tf:
            arr = get_term_frequency(filepath,doc,stop_words,cordid_to_pmc)
            train_doc_tf[doc] = arr
        
    return train_doc_tf

def collection_frequency(cordid_to_pmc,train_doc_tf):
    
    vocab = {}
    
    for doc in train_doc_tf:
        arr = train_doc_tf[doc]

        for word in arr:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word]+=1
    
    return vocab

def idf(cordid_to_pmc,train_doc_tf):

    idf = {}
    
    for doc in train_doc_tf:
        arr = train_doc_tf[doc]
        arr = set(arr)

        for word in arr:
            if word not in idf:
                idf[word] = 1
            else:
                idf[word]+=1
    
    return idf

def processtop100(top100):
    
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

def prob_term_given_doc(filepath,term,doc,collection_term_freq,doc_size,train_doc_tf,temp):

    ans = 0
    if (term in train_doc_tf[doc]):
        if (term in collection_term_freq):
            freq_in_doc = train_doc_tf[doc][term]
            prob_in_collection = collection_term_freq[term]*temp
            d_j = doc_size[doc] + temp
            ans = (freq_in_doc+prob_in_collection)/d_j

    return ans

def prob_word(word,vocab,doc_size,train_doc_tf,u):
    
    p_w_m = 0
    for doc in train_doc_tf:
        p_w_d = prob_term_given_doc(collection_dir,word,doc,vocab,doc_size,train_doc_tf,u)
        p_w_m += p_w_d
        
    return (p_w_m/len(train_doc_tf))    

def prob_query_term_given_doc(filepath,term,doc,collection_term_freq,doc_size,term_freq_query,u):
    
    ans = 0
    if (term in term_freq_query):
        if (term in collection_term_freq):
            freq_in_doc = term_freq_query[term]
            prob_in_collection = collection_term_freq[term]
            d_j = doc_size[doc]
            ans = (freq_in_doc+u*prob_in_collection)/(d_j+u)
    return ans

def R1(train_doc_tf,word,vocab,all_queries,cordid_to_pmc,stop_words,doc_size,queryno,temp):
    
    p_w_m = 0
    term_freq_query = get_term_frequency_query(all_queries[queryno],stop_words)
    
    for doc in train_doc_tf:
        p_q_m = 1
        p_w_d = prob_term_given_doc(collection_dir,word,doc,vocab,doc_size,train_doc_tf,temp)
        for term in term_freq_query:
            for i in range (0,term_freq_query[term]):
                p_q_d = prob_term_given_doc(collection_dir,term,doc,vocab,doc_size,train_doc_tf,temp)
                # p_q_d = prob_query_term_given_doc(collection_dir,term,doc,vocab,doc_size,term_freq_query,u)
                p_q_m *= p_q_d
        p_w_m += (p_w_d*p_q_m)
    
    return p_w_m/100

def prob_doc_given_word(word,doc,vocab,doc_size,train_doc_tf,temp):
    
    p_w_m = prob_term_given_doc(collection_dir,word,doc,vocab,doc_size,train_doc_tf,temp)
    p_m = 1/100
    
    return (p_w_m*p_m)

def R2(train_doc_tf,word,vocab,all_queries,cordid_to_pmc,stop_words,doc_size,queryno,temp,term_freq_query):
    
    p_w = 1
    ans = 1
    for term in term_freq_query:
        for i in range (0,term_freq_query[term]):
            res = 0
            for doc in train_doc_tf:
                p_m_w = prob_doc_given_word(word,doc,vocab,doc_size,train_doc_tf,temp)
                # p_q_m = prob_query_term_given_doc(collection_dir,term,doc,vocab,doc_size,term_freq_query,u)
                p_q_m = prob_term_given_doc(collection_dir,term,doc,vocab,doc_size,train_doc_tf,temp)
                res += p_m_w*p_q_m
            ans*=res
            
    # ans = ans*p_w*p_w
    
    return ans

def get_expansion_terms(r,vocab,queryno,train_doc_tf,all_queries,cordid_to_pmc,stop_words,doc_size,temp,term_freq_query):

    if (r=="rm1"):
        score = []
        score_dict = {}
        for word in vocab:
            val = R1(train_doc_tf,word,vocab,all_queries,cordid_to_pmc,stop_words,doc_size,queryno,temp)
            score.append(val)
            score_dict[val] = word
    else:
        score = []
        score_dict = {}
        for word in vocab:
            val = R2(train_doc_tf,word,vocab,all_queries,cordid_to_pmc,stop_words,doc_size,queryno,temp,term_freq_query)
            score.append(val)
            score_dict[val] = word

    score = sorted(score,reverse=True)
    expand_terms = []
    if (len(score)>=20):
        end = 20
    else:
        end = len(score)

    k = 0
    for i in range (0,len(score)):
        val = score[i]
        term = score_dict[val]
        if (term not in expand_terms):
            k+=1
            expand_terms.append(term)
        if (k==end):
            break

    return expand_terms

def expand_query(expansions,term_frequency):

    for word in expansions:
        if word in term_frequency:
            term_frequency[word]+=1
        else:
            term_frequency[word]=1

    return term_frequency

def get_vocab_vector(doc_freq_dict):
    i = 0
    vocab = {}

    for word in doc_freq_dict:
        vocab[word] = i
        i+=1
    return vocab

def get_tf(tf):
    
    normalised_tf = 1+math.log(tf,2)
    return normalised_tf

def get_idf(doc_freq_dict,term,total_doc):

    temp = 1 + (total_doc)/(doc_freq_dict[term])
    normalised_idf = math.log(temp,2)
    
    return normalised_idf

def get_query_vector(query,vocab,doc_freq_dict,total_doc):
    
    arr = query
    vector = np.zeros(len(vocab))
    i = 0

    for word in arr:
        if word in doc_freq_dict:
            pos = vocab[word]
            vector[pos] = get_tf(arr[word])*get_idf(doc_freq_dict,word,total_doc)
            
    return vector

def get_doc_vector_train(docid,vocab,doc_freq_dict,total_doc,train_doc_tf):
    
    arr = train_doc_tf[docid]
    vector = np.zeros(len(doc_freq_dict))

    for word in arr:
        if word in vocab:
            pos = vocab[word]
            vector[pos] = get_tf(arr[word])*get_idf(doc_freq_dict,word,total_doc)
            
    return vector

def get_sim(doc_vec,query_vec):
    
    dot_prod = np.dot(doc_vec,query_vec)
    norm_doc_vec = LNG.norm(doc_vec)
    norm_query_vec = LNG.norm(query_vec)
    
    sim = dot_prod/((norm_doc_vec)*(norm_query_vec))
    
    return sim

def get_doc_vector_sum(filepath,queryno,query_relevant,vocab,doc_freq_dict,total_doc,train_doc_tf,dict_rels):
    
    d_r = 0
    d_n = 0
    r = 0
    nr = 0
    # loop = tqdm(query_relevant[queryno], position=0, leave=True)
    arr = query_relevant[queryno]
    doc_index = {}
    
    for doc in arr:

        d_i = get_doc_vector_train(doc,vocab,doc_freq_dict,total_doc,train_doc_tf)
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

def calc_new_ranks(filepath,formulated_query,query_relevant,queryno,doc_freq_dict,total_doc,vocab,train_doc_tf):
    
    new_ranks = {}
    all_ranks = []
    top_docs = query_relevant[queryno]
    # loop = tqdm(top_docs, position=0, leave=True)
    for doc in top_docs:
        d_i = get_doc_vector_train(doc,vocab,doc_freq_dict,total_doc,train_doc_tf)
        rank_val = get_sim(formulated_query,d_i)
        # print(rank_val)
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

def calc_current_score(query_relevant,queryno,rel_check):

    ranklist = query_relevant[queryno]
    dict_rels = get_dict_rels(query_relevant,queryno)
    score = calc_ndcg(ranklist,rel_check,queryno,dict_rels,query_relevant)
    
    return score

def grid_search(rel_check,query_relevant,r):

    k = -0.3
    ndcg = 0
    res = k
    while (k<=1):

        ans = calc_avg_ndcg(rel_check,query_relevant,r,k)
        if (ans>ndcg):

            res = k
            ndcg = ans

        # print(k)
        # print(ans)
        k+=0.1

    return res,ndcg

def calc_avg_ndcg(rel_check,query_relevant,r,u):

    count_queries = len(query_relevant)
    val = 0
    avg = 0
    for q in query_relevant:
        # print(q)
        train_doc_tf = get_doc_tf(collection_dir,query_relevant,cordid_to_pmc,q)
        vocab = collection_frequency(cordid_to_pmc,train_doc_tf)
        doc_freq = idf(cordid_to_pmc,train_doc_tf)
        vocab_vector = get_vocab_vector(doc_freq)
        doc_size = get_all_doc_size(collection_dir,cordid_to_pmc,query_relevant[q],stop_words)
        term_freq_query = get_term_frequency_query(all_queries[q],stop_words)
        expand = get_expansion_terms(r,vocab_vector,q,train_doc_tf,all_queries,cordid_to_pmc,stop_words,doc_size,u,term_freq_query)
        new_query = expand_query(expand,term_freq_query)
        q_vec = get_query_vector(new_query,vocab_vector,doc_freq,len(train_doc_tf))
        dict_rels = get_dict_rels(query_relevant,q)
        d_r,d_n,doc_index = get_doc_vector_sum(collection_dir,q,query_relevant,vocab_vector,doc_freq,len(train_doc_tf),train_doc_tf,dict_rels)
        q_new = q_vec
        new_rank_list,new_rank_dict = calc_new_ranks(collection_dir,q_new,query_relevant,q,doc_freq,len(train_doc_tf),vocab,train_doc_tf)
        ranklist = calc_ranklist(new_rank_list,new_rank_dict)
        res = calc_ndcg(ranklist,rel_check,q,dict_rels,query_relevant)
        val += res

    a = val/count_queries
    
    return a

if __name__ == '__main__':
    
    rm = sys.argv[1]
    query_file = sys.argv[2]
    top_100_file = sys.argv[3]
    collection_dir = sys.argv[4]
    output_file = sys.argv[5]
    expansions_file = sys.argv[6]

    stop_words = get_stopwords()
    cordid_to_pmc = get_doc_freq(collection_dir)
    all_queries = parse_query_doc(query_file,"query")
    query_relevant = processtop100(top_100_file)
    r = rm
    o = open(output_file,"w")
    e = open(expansions_file,"w",encoding="utf-8")

    for q in query_relevant:
        # print(q)
        temp = 1
        train_doc_tf = get_doc_tf(collection_dir,query_relevant,cordid_to_pmc,q)
        vocab = collection_frequency(cordid_to_pmc,train_doc_tf)
        doc_freq = idf(cordid_to_pmc,train_doc_tf)
        vocab_vector = get_vocab_vector(doc_freq)
        doc_size = get_all_doc_size(collection_dir,cordid_to_pmc,query_relevant[q],stop_words)
        temp_u = 1/(get_avg_doc_size(doc_size)*1.2)
        term_freq_query = get_term_frequency_query(all_queries[q],stop_words)
        expand = get_expansion_terms(r,vocab_vector,q,train_doc_tf,all_queries,cordid_to_pmc,stop_words,doc_size,temp_u,term_freq_query)
        e.write(q + " : ")
        
        for i in range (0,len(expand)):
            if (i!=len(expand)-1):
                e.write(expand[i])
                e.write(",")
            else:
                e.write(expand[i])

        e.write("\n")
        new_query = expand_query(expand,term_freq_query)
        q_vec = get_query_vector(new_query,vocab_vector,doc_freq,len(train_doc_tf))
        q_new = q_vec
        new_rank_list,new_rank_dict = calc_new_ranks(collection_dir,q_new,query_relevant,q,doc_freq,len(train_doc_tf),vocab_vector,train_doc_tf)
        ranklist = calc_ranklist(new_rank_list,new_rank_dict)

        for rank in ranklist:
            o.write(q+" ")
            o.write("Q0 ")
            o.write(str(rank) + " ")
            o.write(str(temp) + " ")
            o.write(str(ranklist[rank]) + " ")
            o.write("runid1")
            o.write("\n")
            temp+=1
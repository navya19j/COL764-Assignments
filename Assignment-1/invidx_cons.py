from PorterStemmer import *
import os
import snappy
import re
import json
from bs4 import BeautifulSoup as bs
from collections import OrderedDict

def check_stopword(word,stopwords):
    
    if (stopwords.get(word) is not None):
        return True
    else:
        return False

def conv_num_to_binary(num):

    return "{0:b}".format(num)


def preprocess(all_files,required_tags,docidentifier,stopwords,filepath):
    
    inv_dict = {}
    doc_ids_map = {}
    n = 1
    loop = all_files
    p = PorterStemmer()

    for f in loop:

        file_path = os.path.join(filepath,f)

        with open(file_path,"r") as f:

            content = f.read()
            bs_content = bs(content,'html.parser')
            all_doc = bs_content.find_all('doc')

            for doc in all_doc:

                docno = doc.find(docidentifier).get_text()
                doc_ids_map[n] = docno.strip()
                if_found = {}

                for tags in required_tags:
                    text_content = doc.find_all(str(tags))

                    for text in text_content:
                        doc_content = text.get_text()
                        tokenized_content = re.split(r'''[ ',.:();{}`"\n]''',doc_content)

                        for token in tokenized_content:
                            token = token.lower()
                            if (len(token)>0 and check_stopword(token,stopwords)==False):

                                stem_token = p.stem(token,0,len(token)-1)

                                if inv_dict.get(stem_token) is None:
                                    inv_dict[stem_token] = [n]
                                    if_found[stem_token] = 1

                                else:
                                    # optimise
                                    if if_found.get(stem_token) is None:
                                        inv_dict[stem_token].append(n)
                                        if_found[stem_token] = 1

                n += 1
  
    return (inv_dict,doc_ids_map)

def get_posting_lists(inv_dict):
    posting = []
    inv_ind_dict = {}

    for i in inv_dict:
        k = len(posting)
        m = 0
        for j in inv_dict[i]:
            m += 1
            posting.append(int(j))
            
        inv_ind_dict[i] = [k,m]
            
    return posting,inv_ind_dict

def conv_bin(postings):
    res = []
    for i in range (0,len(postings)):
        res.append(conv_num_to_binary(postings[i]))
    return res

def get_list_to_binary_string(p,filename):
    res = conv_bin(p)
    ans = str(''.join(map(str, res)))
    ans = bytes(ans, 'utf-8')

    with open(filename, 'wb') as f:
        f.write(ans)

def c0_encode(data):

    res = str(conv_num_to_binary(data))
    res = res.zfill(32)

    return str(res)

def c0_compression(postings,inv_ind):
    
    start = 0
    for k in inv_ind:
        for i in range (inv_ind[k][0],inv_ind[k][0]+inv_ind[k][1]):
            postings[i] = c0_encode(postings[i])
        new_len = inv_ind[k][1]*32
        inv_ind[k] = [start,new_len]
        start += new_len
    return postings

def c0_decode(data):
    ans = []
    for i in range (0,len(data),32):
        ans.append(str_to_binary(str(data[i:i+32])))

    return ans

def c1_encode(data):
    
    data_length = data.bit_length()
    mask = (1<<7)-1
    temp = ""
    j = int(data_length/7)+1
    while (data!=0):

        ans = data & mask
        res = str(conv_num_to_binary(ans))
        res = res.zfill(7)

        if (j==int(data_length/7)+1):
            res = res.zfill(8)
            j=j-1
        else:
            res = "1" + res
            
        temp = res + temp
        data = data >> 7
        
    return temp  

def diff(postings,inv_ind):

    for word in inv_ind:
        start = inv_ind[word][0]
        end = start + inv_ind[word][1]
        f = postings[start]
        for i in range (start+1,end):
            temp = postings[i]
            postings[i] -= f
            f = temp
    return postings

def c1_compression(postings,inv_ind):
    
    start = 0
    for k in inv_ind:
        new_len = 0
        for i in range (inv_ind[k][0],inv_ind[k][0]+inv_ind[k][1]):
            postings[i] = c1_encode(postings[i])
            new_len += len(postings[i])
        inv_ind[k] = [start,new_len]
        start += new_len
    return postings

def str_to_binary(num):
    
    return int(num,2)

def write_to_file_compressed(idx,filename):

    ans = str(''.join(map(str, idx)))
    to_add = (8-len(ans)%8)%8

    for i in range (0,to_add):
        ans = ans + "0"

    with open(filename, 'wb') as f:
        temp1 = (int)(len(ans)/8)
        res = int(ans,2)
        res = res.to_bytes(temp1,'big')
        f.write(res)


    # print(end-start)
    return to_add

def c1_decode(data):
    res = ""
    ans = []
    for i in range (0,len(data),8):
        if (data[i] == "0"):
            res += data[i+1:i+8]
            ans.append(str_to_binary(res))
            res = ""
        else:
            res += data[i+1:i+8]

    return ans

def rev_gap(arr,inv_ind):

    for word in inv_ind:
        start = inv_ind[word][0]
        end = inv_ind[word][1] + start

        for i in range (start+1,end):
            arr[i] += arr[i-1]

    return arr

def rev_gap_single(arr):

    for i in range (1,len(arr)):
        arr[i] += arr[i-1]
    return arr
        
    
def c1_decoding(data,start,length):

    # f = open(filename,)
    # data = json.load(f)
    ans = c1_decode(data[start:start+length])
    # ans = rev_gap(ans)

    return ans

def lsb(a,b):
    mask = (1<<b)-1
    ans = a & mask
    res = str(conv_num_to_binary(ans))
    if (len(res)!=b):
        res = res.zfill(b)
    
    return res

def l(x):

    return len(bin(x))-2

def U(l):
    
    res = ""
    for i in range (0,l-1):
        res += "1"
    res += "0"
    
    return res

def c2_encode(data):
    
    res = ""

    t_1 = l(data)
    t_2 = l(t_1)
    
    res+= U(t_2)

    res+= lsb(t_1,t_2-1)

    res+= lsb(data,t_1-1)

    return res

def c2_decode(data):
    i = 0
    start = True
    ans = []
    cnt = 0
            
    while (i<len(data)):

        if (start and i+3 <= len(data) and data[i]=="0" and data[i:i+3]=="000"):
            ans.append(1)
            i = i+3
                
        elif (i<len(data) and start!=True and data[i]=="0"):
            l2 = cnt+1
            l1 = str_to_binary("1" + data[i+1:i+l2])
            x = str_to_binary("1"+ data[i+l2:i+l2+l1-1])
            i = i+l2+l1-1
            ans.append(x)
            start = True
            cnt = 0
            
        elif (i < len(data) and data[i]=="1"):
            cnt+=1
            start = False
            i+=1

    return ans

def c2_compression(postings,inv_ind):
    
    start = 0
    for k in inv_ind:
        new_len = 0
        for i in range (inv_ind[k][0],inv_ind[k][0]+inv_ind[k][1]):
            postings[i] = c2_encode(postings[i])
            new_len += len(postings[i])
        inv_ind[k][0] = start
        inv_ind[k][1] = new_len
        start += new_len
        
    return postings

def c2_decoding(filename,start,length):
    
    f = open(filename,)
    data = json.load(f)
    ans = c2_decode((str(data))[start:start+length])
    rev_gap(ans)

    return ans

def c3_compression(data,inv_ind):
    
    res = str('.'.join(map(str, postings)))
    r = snappy.compress(res)

    return r

def write_to_file_c3(r,filename):

    with open(filename, 'wb') as binary_file:
        binary_file.write(r)

def c3_decode(data):
    
    data = snappy.uncompress(data)
    data = data.decode()
    ans = data.split(".")
    ans = [int(i) for i in ans]
    
    return ans

def c3_decoding(filename,inv_ind):

    f = open(filename,"rb")

    data = f.read()
    data = snappy.uncompress(data)
    out = data.decode()
    ans = c1_decode(out)

    for k in inv_ind:
        rev_gap(ans[inv_ind[k][0]:inv_ind[k][0]+inv_ind[k][1]])

    return ans

def get_stopwords_from_file():

    stopwordfile = open(sys.argv[3])
    stopwords = []
    for x in stopwordfile:
        stopwords.append(x)

    stopword_hash = {}
    for y in stopwords:
        stopword_hash[y] = 1

    stopwordfile.close()

    return stopword_hash

def get_xml_tags():
    
    xml_tags = open(sys.argv[5])
    required_tags = []
    all_lines = xml_tags.readlines()

    doc_identifier = all_lines[0].lower().strip()
    required_tags = list(map(lambda x : x.strip(),all_lines[1:]))
    required_tags = list(map(lambda x : x.lower(),required_tags))
    
    return (doc_identifier,required_tags)

def create_posting_list(postings,comp,inv_ind,doc_ids_map,stopwords):

    filepath = sys.argv[2] +".idx"
    if (comp == 0):
        postings = c0_compression(postings,inv_ind)
        skip = write_to_file_compressed(postings,filepath)
    
    elif (comp == 1):
        ans = diff(postings,inv_ind)
        ans = c1_compression(ans,inv_ind)
        skip = write_to_file_compressed(ans,filepath)

    elif (comp == 2):
        ans = diff(postings,inv_ind)
        ans = c2_compression(ans,inv_ind)
        skip = write_to_file_compressed(ans,filepath)
    
    elif (comp == 3):
        ans = diff(postings,inv_ind)
        res = c3_compression(ans,inv_ind)
        write_to_file_c3(res,filepath)

    elif (comp == 4):
        print("not implemented")

    elif (comp == 5):
        print("not implemented")
        
    with open(sys.argv[2] + ".dict", 'w') as file:
        if (comp!=3):
            res = {"inv": inv_ind,"map": doc_ids_map,"compression": comp, "stopwords":stopwords, "to_skip":skip}
        else:
            res = {"inv": inv_ind,"map": doc_ids_map,"compression": comp, "stopwords":stopwords}
        json.dump(res, file)

if __name__ == "__main__":

    all_files = list(os.listdir(sys.argv[1]))
    filepath = sys.argv[1]
    stopwords = get_stopwords_from_file()
    doc_identifier = get_xml_tags()[0]
    required_tags = get_xml_tags()[1]

    all_parsed_files = preprocess(all_files,required_tags,doc_identifier,stopwords,filepath)
    inv_dict = all_parsed_files[0]
    doc_ids_map = all_parsed_files[1]

    inv_ind = get_posting_lists(inv_dict)[1]
    postings = get_posting_lists(inv_dict)[0]

    create_posting_list(postings,int(sys.argv[4]),inv_ind,doc_ids_map,stopwords)
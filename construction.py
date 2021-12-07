import pickle
from os.path import isfile, join
from collections import Counter
import pandas as pd
class Construction:
    def __init__(self,docs_ls_file,word_freq_file):
        if not isfile(join("./",docs_ls_file)):
            self.docs_ls_scv = pd.Series(dtype="object")
            self.word_freq = Counter()
        else:
            self.docs_ls_csv = pd.read_csv(docs_ls_file)["0"]    
            
            with open(word_freq_file,"rb") as f:
                self.word_freq = pickle.load(f)
        
        self.freq_ls = None
        self.index_word_dic = None
        self.word_index_dic = None
        
    def add_docs_ls(self,docs_ls,docs_ls_file,word_freq_file):
        concate_docs_ls = pd.concat([self.docs_ls_csv,docs_ls])
        self.word_freq += Counter(sum(docs_ls.tolist(),[]))
        
        concate_docs_ls.to_csv(docs_ls_file,index=False)
        
        self.docs_ls_csv = pd.read_csv(docs_ls_file)["0"]
        with open(word_freq_file,'wb') as f:
            pickle.dump(self.word_freq, f)

    def get_word_freq_ls(self,vocab_size):
        word_ls,self.freq_ls = zip(*self.word_freq.most_common()[:vocab_size])
        self.index_word_dic = dict(enumerate(word_ls))
        self.word_index_dic = {}
        for i,w in enumerate(word_ls):
            self.word_index_dic[w] = i
        with open("freq_ls.pickle","wb") as f:
            pickle.dump(self.freq_ls,f)
        with open("index_word_dic.pickle","wb") as f:
            pickle.dump(self.index_word_dic,f)
        with open("word_index_dic.pickle","wb") as f:
            pickle.dump(self.word_index_dic,f)
            
    def set_docs_ls(self):
        # 문서수 : 6041696
        for i in range(-(-6041696//10000)):
            with open(f"docs_ls/docs_ls{i}","wb") as f:
                pickle.dump(self.docs_ls_csv[i*10000:(i+1)*10000].str.replace("[","").str.replace("]","").str.replace("'","").str.replace(" ","").str.split(","), f)


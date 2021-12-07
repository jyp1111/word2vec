import pickle
from os.path import isfile, join
import numpy as np
class Skip_gram:
    def __init__(self,docs_ls_file,window_size):
        with open("freq_ls.pickle","rb") as f:
            self.freq_ls=pickle.load(f)
        with open("index_word_dic.pickle","rb") as f:
            self.index_word_dic=pickle.load(f)
        with open("word_index_dic.pickle","rb") as f:
            self.word_index_dic=pickle.load(f)
        with open(docs_ls_file,"rb") as f:
            self.docs_ls=pickle.load(f)

        self.window_size=window_size
        if not isfile(join("./","coef.pickle")):
            self.coef_=None
        else:  
            with open("coef.pickle","rb") as f:
                self.coef_=pickle.load(f)        

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def negative_sampling(self):
        n=len(self.word_index_dic)
        fre_ls=np.array(list(self.freq_ls))
        distribution=np.power(fre_ls,0.75)/np.power(fre_ls,0.75).sum()
        return np.random.choice(range(n), size=20, p=distribution)

    
    def connection(self,doc,i):
        result=set()
        for j,v in enumerate(doc):
            if v==self.index_word_dic[i]:
                result=result.union({self.word_index_dic[doc[k]] for k in range(max(j-self.window_size,0),min(j+self.window_size+1,len(doc))) if doc[k] in self.word_index_dic})
        return result-{i}            

    def train(self,vec_dim,learning_rate):
        n=len(self.word_index_dic)
        m=len(self.docs_ls)
        
        if self.coef_==None:
            W_1=np.random.rand(n,vec_dim)
            W_2=np.random.rand(n,vec_dim)
        else: 
            W_1,W_2=self.coef_

        for i in range(n):
            return_set=set().union(*self.docs_ls.apply(self.connection,args=(i,)))

            neighbor=sorted(list(return_set))
            negative_words=self.negative_sampling()

            positive=np.ones(len(neighbor))
            negative=np.zeros(20)

            y=np.concatenate((positive,negative))
            relative_indices=neighbor+list(negative_words)
            hyp=self.sigmoid(np.dot(W_2[relative_indices],W_1[i]))
            loss=-(y*np.log(hyp)+(1-y)*np.log(1-hyp)).mean()

            first_par=W_1[i]
            W_1[i]-=np.dot(W_2[relative_indices].T,hyp-y)*learning_rate/len(y)
            W_2[relative_indices]-=np.dot((hyp-y).reshape(-1,1),first_par.reshape(1,-1))*learning_rate/len(y)
            if i%100==0:
                print(i)
        self.coef_=(W_1,W_2)
        with open("coef.pickle","wb") as f:
            pickle.dump(self.coef_,f)
        
    def check(self,word1,word2,word3):
        word_matrix=self.coef_[0]
        
        word1_index=self.word_index_dic[word1]
        word2_index=self.word_index_dic[word2]
        word3_index=self.word_index_dic[word3]
        result_vec=word_matrix[word1_index]-word_matrix[word2_index]+word_matrix[word3_index]
        
        return self.index_word_dic[np.argmin(np.multiply(word_matrix-result_vec,word_matrix-result_vec).sum(axis=1))]
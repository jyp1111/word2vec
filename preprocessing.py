import pandas as pd
from os.path import isfile, join
from os import listdir
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

if __name__ == "__main__":
    with open("wiki_00",mode="r",encoding="utf-8") as f:
        text = f.read()
        
    textdf = pd.DataFrame(text.split("</doc>")[:-1])
    textdf.columns = ["content"]
    textdf["content"] = textdf["content"].str.replace('<.*?>',"").str.replace("\n"," ")
    textdf["tokenize"] = textdf["content"].apply(word_tokenize)
    textdf["pos_tag"] = textdf["tokenize"].apply(pos_tag)

    pos = ["JJ","JJR", "JJS","NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"]

    textdf["reduced_postag"] = textdf["pos_tag"].apply(lambda ls: [v for v in ls if (v[1] in pos) and (len(v[0])>1)])
    textdf["token_ls"] = textdf["reduced_postag"].apply(lambda ls: [v[0] for v in ls])

    df = pd.DataFrame([",".join(token_ls) for token_ls in textdf["token_ls"].tolist()])

    csv_files=["./text_csv/"+f  for f in listdir("./text_csv/")  if isfile(join("./text_csv/", f))]
    csv_files.sort()
    for x in csv_files:
        df = pd.read_csv(x)
        df.dropna(inplace=True)
        df["0"] = df["0"].apply(lambda v : ",".join(v.split(",")[1:]))
        df.dropna(inplace=True)
        df["0"].to_csv(x,index=False)

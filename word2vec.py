import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
from construction import Construction
from skip_gram import Skip_gram
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_construction",
                        default=False,
                        action='store_true',
                        help="Input corpus")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run test")
    parser.add_argument("--docs_ls_file",
                        default=None,
                        type=str,
                        help="The input data file")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=2e-5,
                        help="learning rate in each group")
    parser.add_argument('--vector_dim', 
                        type=int, 
                        default=3,
                        help="vector dimension")
    parser.add_argument('--window_size', 
                        type=int, 
                        default=1,
                        help="window_size")

    args = parser.parse_args()
    if args.do_construction:
        csv_files=["./text_csv/"+f  for f in listdir("./text_csv/")  if isfile(join("./text_csv/", f))]
        csv_files.sort()
        wiki = Construction("docs_ls.csv","freq.pickle")
        for file in csv_files[135:]:
            df = pd.read_csv(file)
            wiki.add_docs_ls(df["0"].str.split(","),"docs_ls.csv","freq.pickle")
        wiki.get_word_freq_ls(30000)
        wiki.set_docs_ls()
    else:   
        model=Skip_gram(args.docs_ls_file,args.window_size)
        if args.do_train:
            model.train(args.vector_dim,args.learning_rate)
        if args.do_test:
            model.check("Athens","Greece","Korea")
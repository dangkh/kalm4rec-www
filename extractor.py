import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import shutil
from kwExtractorHelper.utils import *

CITIES_LIST = ['charlotte', 'edinburgh', 'lasvegas', 'london', 'phoenix', 'pittsburgh', 'singapore', 'tripadvisor']


parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='edinburgh', help=f'choose city{CITIES_LIST}')
parser.add_argument('--edgeType', type=str, default='IUF', help='IUF or IRF')
parser.add_argument('--kwExtractor', type=str, default='kw_spacy', help='kw_spacy, kw_NLTK, kw_RAKE, kw_YAKE')
args = parser.parse_args()

print("args:", args)
if args.kwExtractor not in ["kw_spacy", "kw_NLTK", "kw_RAKE", "kw_YAKE"]:
    raise "extractor not in list"
city = args.city
sets = ['train', 'test']
dt = pd.read_csv('./data/reviews/{}.csv'.format(city))
for setname in sets:
    print("Processing for", city, setname)
    uids = load_split(city=city, setname=setname)
    dt_set = dt[dt['user_id'].isin(uids)]
    odir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/' + setname
    mkdir(odir)
    extract_raw_keywords_for_reviews(dt_set, ofile=os.path.join(odir, city + '-keywords.json'), keep=['ADJ', 'NOUN', 'PROPN', 'VERB'],
                                    overwrite=False, review2keyword_ofile=os.path.join(odir,city+"-review2keywords.csv"),
                                    argsExtractor = args.kwExtractor)

min_freq = 3
city = args.city
print("Processing Filter min frequency 3 for", city)
ifile = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/train/{}-keywords.json'.format(city)
ofile = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_{}/train/{}-keywords.json'.format(min_freq, city)
mkdir('./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_{}/train/'.format(min_freq))
filter_keywords(ifile, ofile, min_freq=min_freq)


if args.edgeType == "IRF":
    names = ['train']
    for setname in names:
        # idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/' + setname
        idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/' + setname
        odir = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq/' + setname
        mkdir(odir)
        for fname in os.listdir(idir):
            if fname.startswith('.') or not fname.endswith(".json"):
                continue
            print("Processing for", fname)
            ifile = os.path.join(idir, fname)
            ofile = os.path.join(odir, fname)
            group_keywords_for_users(ifile, ofile)
        print("------------")


    # idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/train'
    idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train'
    odir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IRF'
    mkdir(odir)
    compute_irf_for_dir(idir, odir, N=1000)


    irf_dir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IRF'
    idir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq'
    odir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/tf_irf'

    city2irf = {}

    setname = "train"
    idir = os.path.join(idir_root, setname)
    odir = os.path.join(odir_root, setname)
    mkdir(odir)
    fname = f"{city}-keywords.json"
    ifile = os.path.join(idir, fname)
    ofile = os.path.join(odir, fname)
    print("Processing for", ifile)
    compute_tfirf(ifile, ofile, irf=get_irf(fname, city2irf, irf_dir))
else:

    names = ['train']
    for setname in names:
        # idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/' + setname
        idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/' + setname
        odir = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq/' + setname
        mkdir(odir)
        for fname in os.listdir(idir):
            if fname.startswith('.') or not fname.endswith(".json"):
                continue
            print("Processing for", fname)
            ifile = os.path.join(idir, fname)
            ofile = os.path.join(odir, fname)
            group_keywords_for_rests(ifile, ofile)
        print("------------")


    idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train'
    odir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IUF'
    mkdir(odir)
    compute_iUf_for_dir(idir, odir, N=1000)

    iuf_dir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IUF'
    idir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq'
    odir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/tf_iuf'

    city2iuf = {}

    setname = "train"
    idir = os.path.join(idir_root, setname)
    odir = os.path.join(odir_root, setname)
    mkdir(odir)
    fname = f"{city}-keywords.json"
    ifile = os.path.join(idir, fname)
    ofile = os.path.join(odir, fname)
    print("Processing for", ifile)
    compute_tfiuf(ifile, ofile, irf=get_irf(fname, city2iuf, iuf_dir))


model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('whaleloops/phrase-bert')

typeFile = ['train', 'test']
city = args.city
mkdir("./data/keywords/")
mkdir("./data/score/")
mkdir("./data/embedding/")
for tp in typeFile:
    if tp == 'train':
        f = open(f'./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/{tp}/{city}-keywords.json')
    else:
        f = open(f'./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/{tp}/{city}-keywords.json')
    data = json.load(f)
    keys = [ii for ii in data]
    kwEB = []
    kwL = []
    kws = [kw for kw in data[keys[0]]]
    print("Encoding")
    inputs = model.encode(kws)
    kwEB_pad = np.asarray(inputs)
    np.save(f'./data/embedding/{city}_kwSenEB_pad_{tp}.npy', kwEB_pad)
    f.close()

# Download train: train is filtered file at ./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train) then rename as: {city}-keyword_train.json
# Download test: train is filtered file at ./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/test) then rename as: {city}-keyword_test.json
# move those train/test keywords file to ./data/keywords directory

# Download irf and tf_irf;  rename as {city}-keyword-IRF.json {city}-keyword-TFIRF.json ; move to  ./data/score/{city}-keyword-TFIRF.json
# Download iuf and tf_iuf;  rename as {city}-keyword-IUF.json {city}-keyword-TFIUF.json ; move to  ./data/score/{city}-keyword-TFIUF.json
shutil.move(f"./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/train/{city}-keywords.json",
    f"./data/keywords/{city}-keywords_train.json")
shutil.move(f"./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train/{city}-keywords.json",
    f"./data/keywords/{city}-keywords_train.json")
shutil.move(f"./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/test/{city}-keywords.json",
    f"./data/keywords/{city}-keywords_test.json")
name1 = "IUF"
name2 = "TFIUF"
name3 = "tf_iuf"
if args.edgeType == "IRF":
    name1 = "IRF"
    name2 = "TFIRF"
    name3 = "tf_irf"
shutil.move(f"./data/preprocessed/by_city-users_min_3_reviews/keywords_{name1}/{city}-keywords.json", f"./data/score/{city}-keywords-frequency.json")
shutil.move(f"./data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/{name3}/train/{city}-keywords.json", f"./data/score/{city}-keywords-{name2}.json")


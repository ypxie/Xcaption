import nltk as nl
import matplotlib.pyplot as plt
from json import loads, dump
from os import listdir
from os.path import isfile, join
from random import randint, choice
from itertools import chain
from numpy import mean, std
from copy import copy
import numpy as np
import collections



def LoadCorpus(directory):
    """Returns a list of annotations found in the directory given in the string 'directory'."""
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    corpus = []
    file_name_list = []
    for c_file in files:
        corpus.append(loads(open(directory + c_file).read()))
        file_name_list.append(c_file)
    return corpus, file_name_list

def LoadWords(fname):
    """Returns a dictionary of categories to be passed to GenerateParaphrases(). These categories
    are loaded from a json file with the name 'fname'."""
    json_wordswaps = open(fname).read()
    return loads(json_wordswaps)

def Dict2Sentence(annotation_dict):
    """Converts the values in 'annotation_dict' into a string by concatenating them."""
    annotation_dict_copy = copy(annotation_dict)
    annotation_dict_copy.pop("conclusion")
    sentence = ''
    for i in annotation_dict_copy.values():
        sentence = sentence + i[1].capitalize() + ' '
    return sentence

def LoadTemplates(ftemp):
    """Returns a list of swappable phrases to be passed to GenerateParaphrases"""
    tempfile = open(ftemp, 'r')
    tempfile.next()    # skip the header line
    phrases = []
    for c_line in tempfile:
        c_line_split = c_line.split('\t')
        c_line_0 = c_line_split[0]
        c_line_1 = map(int, c_line_split[1].split(' '))
        c_line_2 = c_line_split[2].translate(None, '\n').split(' ')
        #c_line_2 = map(int, c_line_2) 
        c_line_3 = c_line_split[3].translate(None, '\n').split(' ')
        phrases.append([c_line_0, c_line_1, c_line_2, c_line_3])
    return phrases

def GenerateParaphrases(candidate, phrases, wordswaps, n_paraphrases=10):
    """Pass one candidate at a time."""
    ordered_catetories = ['nuclear_feature', 'nuclear_crowding', 'polarity','mitosis', 'nucleoli', 'conclusion']
    keys_cats = list(wordswaps.keys())
    keys_cats.remove("conclusion")
    categories = {str(wordswaps[i][0]): i for i in keys_cats}
    n_phrases = len(phrases)
    paraphrases = []
    # for now we set it to 1
    for i in range(n_paraphrases):
        c_degrees = {cat: candidate[cat][0] for cat in categories.values()}    # dict with annotation info
        cats_used = []
        sentences = collections.OrderedDict()
        for cat in ordered_catetories: sentences[cat] = '' # list of sentences 
        sentences['conclusion'] = candidate['conclusion'] # keep conclusion the same
        count = 0        # number of sentences
        while not (len(list(c_degrees.keys()))==0) and count < 5:
            # select a category
            cats_unused = list(c_degrees.keys())
            c_cat = choice(cats_unused)        # randomly select unused category
            c_cat_degree = c_degrees[c_cat]    # get degree of category for this file
            
            # selects idxs from phrase that belong to unused category c_cat
            c_phrases_idxs = [i for i in range(n_phrases) if (wordswaps[c_cat][0] in phrases[i][1]) 
                          and (c_cat_degree in phrases[i][2] or phrases[i][2][0]=='')]
            
            # build list of idxs for used categories so they are ignored when selecting phrases
            c_used_cat_idxs = [[i for i in range(n_phrases) if (wordswaps[c_cat_used][0] in phrases[i][1])] 
                                   for c_cat_used in cats_used]
            c_used_cat_idxs = list(chain.from_iterable(c_used_cat_idxs))    # converts iterable to list
            
            # select phrase, its category, and degree from list
            c_phrases_idxs = [i for i in c_phrases_idxs if i not in c_used_cat_idxs]
            c_phrase_selected_idx = choice(c_phrases_idxs)       # randomly select phrase
            c_phrase_text = phrases[c_phrase_selected_idx][0]    # text of paraphrase
            c_phrase_cats = phrases[c_phrase_selected_idx][1]    # idxs of categories
            #c_phrase_degrees = phrases[c_phrase_selected_idx][2] # idxs of degrees
            c_phrase_adverbs = phrases[c_phrase_selected_idx][3] # bool to indicate if adverb

            # replace blank in phrase with degree
            c_categories = [categories[str(i)] for i in c_phrase_cats]
            c_phrase_adverbs_list = [int(j) for i, j in zip(c_phrase_cats, c_phrase_adverbs)]
            c_phrase_cats_degrees = [c_degrees[i] for i in c_categories]
            c_degree_text = [wordswaps[c_category][1][1][c_cat_degree] if is_adv else wordswaps[c_category][1][0][c_cat_degree] 
                             for c_category, c_cat_degree, is_adv in zip(c_categories, c_phrase_cats_degrees, 
                                                                         c_phrase_adverbs_list)]
            for i in c_degree_text:
                c_phrase_text = c_phrase_text.replace(blank, i, 1)

            # bookkeeping for loop
            assert(len(c_categories) == 1) # this assumption is based on that every feature description is seperatable
            sentences[c_categories[0]] = [c_cat_degree, c_phrase_text]    # add new sentence to selected sentences
            [c_degrees.pop(i) for i in c_categories]    # remove used cat & degree from list to prevent selection
            [cats_used.append(i) for i in c_categories] # add category to list of used degrees 
            count += 1

        # capitalize and concatenate list of sentences
        paraphrase = ''
        for i in sentences.keys():
            paraphrase = paraphrase + sentences[i][1].capitalize() 
            sentences[i][1] = sentences[i][1].capitalize()
        # add paraphrased sentences to list to be returned
        # paraphrases.append(paraphrase)
    return paraphrase, sentences

def BleuScore(candidate, references, weights=[0.25, 0.25, 0.25, 0.25]):
    return nl.bleu(references, candidate, weights)

def ModifiedBleuScore(candidate, references, weights=[0.25, 0.25, 0.25, 0.25]):
    n_references = len(references)
    assert n_references > 1
    
    """Returns Bleu score for 'candidate' using the list of 'references' and 'weights'."""
    # compute raw bleu score
    raw_bleu = BleuScore(candidate, references, weights)
    
    # compute bleu score of each reference with the other references
    bleu_refs = []
    for i in range(1, n_references):
        ref_copy = copy(ref_list)
        ref_copy.pop(i)
        bleu_refs.append(BleuScore(ref_list[i], ref_copy, weights))
    
    # bleu score normalized by reference mean and std
    ref_mean = mean(bleu_refs)
    ref_std = std(bleu_refs)
    mod_bleu = (raw_bleu-(ref_mean-raw_bleu-ref_std)**2)/ref_mean
    if (mod_bleu > 1.0):
        mod_bleu = 1
    return mod_bleu
    
# set paths
d_corpus = "../Annotation/"
f_tempfile = "list_descriptions_small.txt"
f_wordswap = "word_swaps.json"
blank = "____"
conclusion = "conclusion"

# load everythin
corpus, corpus_name_list = LoadCorpus(d_corpus)
phrases = LoadTemplates(f_tempfile)
words = LoadWords(f_wordswap)

# random mapping
n_references = 1
save_path = '../data_organized_random/annotation_mapped/'
for i in range(len(corpus)):
    c_data = copy(corpus[i])
    ref_seq, ref_table = GenerateParaphrases(c_data, phrases, words, n_references)
    with open(save_path+corpus_name_list[i], 'w') as outfile:
        dump(ref_table, outfile)
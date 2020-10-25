import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
import heapq


class Segment:

    def __init__(self, Pw, Pwc=None, Pwcc=None, lambda_bigram=0.1, lambda_trigram=0.1):
        self.Pw = Pw
        self.Pwc = Pwc
        self.Pwcc = Pwcc
        self.lambda_bigram = lambda_bigram
        self.lambda_trigram = lambda_trigram

    def segment(self, text, default=False):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        if default:
        	return [w for w in text]
        # initialize the heap
        entries = []
        matches = self.find_matches(text)
        for word in matches:
            e = [word, len(word), -log10(self.Pw(word)), None]
            heapq.heappush(entries, e)

        # iteratively fill in the hash table
        chart = {}
        while len(entries) != 0:
            entry = heapq.heappop(entries)
            endindex = entry[1]
            if endindex in chart:
                if chart[endindex][2] > entry[2]:
                    chart[endindex] = entry
            else:
                chart[endindex] = entry
            # Add more entry candidates to the heap if there are characters left
            if len(text[endindex:]) > 0:
                matches = self.find_matches(text[endindex:])
                for word in matches:
                    new_prob = self.Pw(word)
                    if self.Pwc is not None:
                        new_prob = self.lambda_bigram * self.Pwc((entry[0], word)) + (1 - self.lambda_bigram) * new_prob
                    if self.Pwcc is not None:
                        if entry[-1] is not None:
                            new_prob = self.lambda_trigram * self.Pwcc((chart[entry[-1]][0], entry[0], word)) + (1 - self.lambda_trigram) * new_prob
                        else:
                            new_prob = (1 - self.lambda_trigram) * new_prob
                    new_prob = entry[2] - log10(new_prob)
                    e = [word, endindex+len(word), new_prob, endindex]
                    if (e[1] not in chart) or (chart[e[1]][2] > e[2]):
                        heapq.heappush(entries, e)

        # get the best segmentation
        final_word = chart[len(text)]
        segmentation = [final_word[0]]
        while final_word[-1] is not None:
            final_word = chart[final_word[-1]]
            segmentation = [final_word[0]] + segmentation
        return segmentation

    def find_matches(self, word):
        def check_for_digits(word, max_len):
            result = word[0]
            for i in range(1, len(word)):
                if word[i].isdigit() or word[i] in ['·', '：', '.', ':']:
                    result += word[i]
                elif word[i] in ['月', '日', '年']:
                    result += word[i]
                    break
                else:
                    break
            if len(result) > max_len:
                return True, result
            if not result.isdigit():
                return True, result
            return False, None

        "Return all the matches of word based on word_dic"
        matches = []
        if word[0].isdigit():
            is_digit, digits = check_for_digits(word, 5)
            if is_digit:
                matches.append(digits)
                return matches
        for i in range(1, min(6, len(word)+1)):
            matches.append(word[:i])
        return matches

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        special_chars = ['·', '万余', '：', '.', ':']
        new_key = key
        for char in special_chars:
           new_key = new_key.replace(char, '')
        if key in self: return self[key]/self.N 
        elif new_key.isdigit(): return 1./self.N
        else: return self.missingfn(key, self.N)

class Pdist_cond(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self,  Pw, data=[]):
        self.Pw = Pw
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)

    def __call__(self, key): 
        if key in self:
            return self[key]/self.Pw[key[0]]  
        else: return 0

class Pdist_cond_tri(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self,  Pwc, data=[]):
        self.Pwc = Pwc
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)

    def __call__(self, key): 
        if key in self:
            return self[key]/self.Pwc[(key[0], key[1])]  
        else: return 0

def datafile(name, sep='\t', mode='unigram'):
    "Read key,value pairs from file."
    with open(name) as fh:
        if mode == 'unigram':
            for line in fh:
                (key, value) = line.split(sep)
                yield (key, value)
        elif mode == 'bigram':
            for line in fh:
                (key1, key2, value) = line.split()
                yield ((key1, key2), value)
        elif mode == 'trigram':
            for line in fh:
                (key1, key2, key3, value) = line.split()
                yield ((key1, key2, key3), value)                
        else:
            raise ValueError('Unrecognized ngram selected')


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-t", "--trigramcounts", dest='counts3w', default=os.path.join('data', 'count_3w.txt'), help="trigram counts [default: data/count_3w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()


    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    missing_func = (lambda k, N: 10./(N * 5500**(len(k))))
    Pw = Pdist(data=datafile(opts.counts1w), missingfn=missing_func)
    Pwc = Pdist_cond(Pw, data=datafile(opts.counts2w, mode='bigram'))
    Pwcc = Pdist_cond_tri(Pwc, data=datafile(opts.counts3w, mode='trigram'))
    segmenter = Segment(Pw, Pwc, Pwcc)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))

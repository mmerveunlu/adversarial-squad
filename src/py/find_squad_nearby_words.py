"""Find nearby words for words in SQuAD questions."""
import argparse
import json
from nltk.tokenize import word_tokenize
import numpy as np
from scipy import spatial
from sklearn.neighbors import KDTree
import string
import sys
from tqdm import tqdm

SQUAD_DEV_FILE = 'data/squad/sample1.json'
PUNCTUATION = set(string.punctuation) | set(['``', "''"])

def parse_args():
  parser = argparse.ArgumentParser('Find nearby words for words in SQuAD questions.')
  parser.add_argument('wordvec_file', help='File with word vectors.')
  parser.add_argument('--input-file', '-f',
                      help=('Input file (defaults to dev file).'),
                      default=SQUAD_DEV_FILE)
  parser.add_argument('--num-neighbors', '-n', type=int, default=1,
                      help='Number of neighbors per word (default = 1).')
  parser.add_argument('--out-file', '-o', help="path to save the result")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def extract_words(input_file):
  """ 
  extracts the set of words from given file
  Args: 
    @input_file: string, the path of the file
  Returns: 
    @words: set of string, contains words
  """ 
  with open(input_file) as f:
    dataset = json.load(f)
  words = set()
  for a in dataset['data']:
    for p in a['paragraphs']:
      for q in p['qas']:
        cur_words = set(w.lower() for w in word_tokenize(q['question'])
                        if w not in PUNCTUATION)
        words |= cur_words
  return words

def get_nearby_words(main_words,wordvec_file,num_neighbors):
  """ 
  finds the nearby words by Glove Vectors for a given list
  Args: 
   @main_words: set of string, words 
   @wordvec_file: string, path of wrod 2 vector files
   @num_neighbors: int, number of close words
  Returns: 
   @nearby_words: dict
  """ 
  main_inds = {}
  all_words = []
  all_vecs = []
  with open(wordvec_file) as f:
    for i, line in tqdm(enumerate(f)):
      toks = line.rstrip().split(' ')
      word = str(toks[0]).encode('utf-8').decode('latin-1')
      vec = np.array([float(x) for x in toks[1:]])
      all_words.append(word)
      all_vecs.append(vec)
      if word in main_words:
        main_inds[word] = i
  print('Found vectors for %d/%d words = %.2f%%' % (
      len(main_inds), len(main_words), 100.0 * len(main_inds) / len(main_words)))
  tree = KDTree(all_vecs)
  nearby_words = {}
  for word in tqdm(main_inds):
    dists, inds = tree.query([all_vecs[main_inds[word]]],
                             k=num_neighbors + 1)
    nearby_words[word] = [
        {'word': all_words[i], 'dist': d} for d, i in zip(dists[0], inds[0])]
  return nearby_words

def main(args):
  words = extract_words(args.input_file)
  print('Found %d words' % len(words))
  nearby_words = get_nearby_words(words,args.wordvec_file,args.num_neighbors)
  with open(args.out_file,"w") as fp:
    json.dump(nearby_words, fp, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  args = parse_args()
  main(args)


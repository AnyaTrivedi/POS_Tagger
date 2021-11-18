# POS Tagger using Hidden Markov Models

## Data
Datasets. There are two datasets under data:
1. Ice cream climatology dataset (ic). This is a synthetic toy dataset, where the observations are the number of ice cream cones eaten each day. The hidden states are the whether each day (C for cold and H for hot). You will use ictrain for training and ictest for testing. Adapted from Jason Eisner’s HMM assignments.
2. English POS tagging dataset (en). This dataset consists of word sequences and the POS tag for each word. Coarse English tags are used-  i.e. only the first character of the original tag is used, e.g., different types of nouns (NN, NNP, NNS) are reduced to simply nouns (N). 

### File format
Each line has a single word/tag pair separated by the / character. Punctuation marks count as words. The special word ### is used for sentence boundaries, and is always tagged with ###. You can think of it as the special start symbol * and the stop symbol STOP too. When you generate ### that represents a decision to end a sentence (so ### = STOP). When you then generate the next tag, conditioning on ### as the previous tag means that you’re at the beginning of a new sentence (so ### = STOP).

## Model
The final model developed in `hmm.py` uses a bigram tagger with backoff and interpolation. Here, tags are learned for each position considering both the word at that position, as well as the tag in the previous position. Set the flags in order to use a vanilla version of HMM (no interpolation or backoff), only interpolation (or backoff), or a combination of the two. Lambda values have been learned in validation.

NOTE: In `vanillaHMM.py`, no interpolation and backoff is used. Thus, the emission probabilities are from tag -> word is P(word|tag)  = count(word, tag)/ count(tag) and the transition probability from tag1 -> tag2 is P(tag2|tag1) = count(tag1, tag2)/ count(tag1). It is used to train and test the ic dataset.

## Evaluation

While accuracy is the main evaluation metric, additional metrics can be used for debugging:

1. Accuracy: percentage of test tokens that received the corret tag, excluding the sentence boundary markers ###.
2. Known Word Accuracy: consider only tokens that appear in the training data.
3. Novel Word Accuracy: consider only tokens that do not appear in the training data, in which case the
model can only use the context to predict the tag.

## Results:
Accuracy using the en-dataset:

1. Backoff and Interpolation: 93.5043009888
2. Backoff Only: 93.472144063
3. Interpolation only: 91.9125331618
4. No Backoff/interpolation (only add-1 smoothning): 91.8241016159

## Running the code
```
python hmm.py --train data/entrain --test data/endev --backoff True --interpolation True --feature_extractor True --verbose True  
```

### Flags:
1. backoff: decides whether to use the smoothened + backoff  probabilities (set to false for the vanilla HMM)
2. interpolation: decides whether to use interpolated transisition probabilities (set to false for the vanilla HMM)
3. feature_extractor: decides whether to use feature extractors for tags (set to false for the vanilla HMM)
4. verbose: decides whether to print the known and unknown acc (used for debugging)

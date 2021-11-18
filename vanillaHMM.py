import numpy as np 
import argparse 
import math 

"""
Vanilla HMM using unsmoothed counts and Viterbi decoding.
"""
#define global matrices

# Keep the unigram counts of the tags and words for calculating matrices
word_counts, tag_counts = {}, {} 
#Word-tag and tag-tag counts
word_tag, tag_tag = {}, {}
#word-tag dictionary. Key: Word, Value: list of tags
WT_dict = {}


def read_file(filename):
    """
        Read the file, unpack into a list of words and tags. 
        Input: Filename to be read from
        Output: Word_List, Tag_Lis
    """
    readline = lambda line: line.strip().split('/')
    words, tags = [], []
    with open(filename) as f:
        for line in f:
            w, t = readline(line)
            words.append(w)
            tags.append(t)

    f.close()

    return words, tags


def counts(filename):
    """ 
        1. Read the training data, and store the word-tag and tag-tag counts
        Input: Filename to be read from
        Output: Update global count variables
    """

    words, tags = read_file(filename)

    #Account for the ### tokens
    #unigram counts of words
    word_counts[words[0]] = 1
    #unigram counts of tags
    tag_counts[tags[0]] = 1

    #word-tag dictionary. Key: Word, Value: list of tags
    WT_dict[words[0]] = [tags[0]]
    #Word-Tag counts
    word_tag[(words[0],tags[0])] = 1

    for i in range(1, len(tags)):
        
        #Increase the unigram counts and the tag-tag occurrences
        word_counts[words[i]] = word_counts.get(words[i],0) + 1
        tag_counts[tags[i]] = tag_counts.get(tags[i],0) + 1
        tag_tag[(tags[i-1], tags[i])] =  tag_tag.get((tags[i-1], tags[i]), 0) + 1

        #For word-tag we may need to add into the dictionary as well.
        #If the (word-tag) pair has not been seen before, check if the word is in the dictionary. 
        #If not, add the tag. If it is in the dictionary, append the tag to the list of possible tags
        if word_tag.get((words[i],tags[i]),0) == 0:
            if words[i] in WT_dict:
                WT_dict[words[i]].append(tags[i])
            else:
                WT_dict[words[i]]= [tags[i]]

        word_tag[(words[i], tags[i])] =  word_tag.get((words[i], tags[i]), 0) + 1

# Make sure to do all computation in the log space.
#Unsmoothened versions
def transition_prob(tag1, tag2):
    """
        Calculate the transition probability from tag1 -> tag2
        P(tag2|tag1) = count(tag1, tag2)/ count(tag1)
    """
    prob = float(tag_tag[(tag1, tag2)])/tag_counts[tag1]
    return math.log(prob)

def emission_prob(tag, word):
    """
        Calculate the emission probability from tag -> word
        P(word|tag)  = count(word, tag)/ count(tag)
    """
    prob = float(word_tag[(word, tag)])/tag_counts[tag]
    return math.log(prob)


def HMM(train_data, test_data):
    """
        2. Compute the emission and transition probabilities in HMM.
        Input: Training data and test data
        Output: Returns transition_matrix, emission_matrix, viterbi matrix, backtracking matrix
    """
    #transition and emission matrices
    transition_matrix = {}
    emission_matrix = {}

    #viterbi matrix: Key: (word pos, tag), value: emission + transitions + old pi values
    pi = {}
    #backtracking martix: Key: (eord pos, tag), value: tag that gave maximum viterbi prob
    p = {}

    test_words, groundtruth_tags = read_file(test_data)
    counts(train_data)

    #UNCOMMENT IF YOU WANT ALL START TAGS TO BE ###
    
    #Initial ### tags: all sentences start with tag ###
    
    # pi[(0, '###')] = 1.0
    # p[(0, '###')] = None

    # #initialize pi and v
    # for tag in WT_dict[test_words[1]]:
    #     # probability =  emission + transisition (calculations in log space)
    #     pi[(1, tag)] =  emission_prob(tag, test_words[1]) + transition_prob('###', tag)
    #     # all first words backpoint to the start tag
    #     p[(1, tag)] = '###'

    # #fill in the tables (follow method by Prof in class - check for every tag preceding all possible tags at currrent pos i)
    # #for each word at current position i
    # for i in range(2, len(test_words)):
    #     #for each posible tag ti at pos i
    #     for ti in WT_dict[test_words[i]]:
    #         #for each tag tj before ti
    #         for tj in WT_dict[test_words[i-1]]:
    #             #check if emission and transition probabilities exist
    #             if (tj, ti) not in transition_matrix:
    #                 transition_matrix[(tj, ti)] = transition_prob(tj, ti)
    #             if (ti, test_words[i]) not in emission_matrix:
    #                 emission_matrix[(ti, test_words[i])] = emission_prob(ti, test_words[i])
                
    #             mu = emission_matrix[(ti, test_words[i])] + transition_matrix[(tj, ti)] + pi[(i-1, tj)]
    #             #find argmax
    #             if mu > pi.get((i, ti), float('-inf')):
    #                 pi[(i, ti)] = mu
    #                 p[(i, ti)] = tj

    # If you dont want all start tags to be ###

    #initialize pi and v
    for tag in WT_dict[test_words[0]]:
        # probability =  emission + transisition (calculations in log space)
        pi[(0, tag)] =  emission_prob(tag, test_words[0]) + transition_prob('###', tag)
        # all first words backpoint to the start tag
        p[(0, tag)] = '###' #should this be None?

    #fill in the tables (follow method by Prof in class - check for every tag preceding all possible tags at currrent pos i)
    #for each word at current position i
    for i in range(1, len(test_words)):
        #for each posible tag ti at pos i
        for ti in WT_dict[test_words[i]]:
            #for each tag tj before ti
            for tj in WT_dict[test_words[i-1]]:
                #check if emission and transition probabilities exist
                if (tj, ti) not in transition_matrix:
                    transition_matrix[(tj, ti)] = transition_prob(tj, ti)
                if (ti, test_words[i]) not in emission_matrix:
                    emission_matrix[(ti, test_words[i])] = emission_prob(ti, test_words[i])
                
                mu = emission_matrix[(ti, test_words[i])] + transition_matrix[(tj, ti)] + pi[(i-1, tj)]
                #find argmax
                if mu > pi.get((i, ti), float('-inf')):
                    pi[(i, ti)] = mu
                    p[(i, ti)] = tj

    return transition_matrix, emission_matrix, pi, p
    

def predictions(test_data, pi, p):
    """
        3. Read the test data and compute the tag sequence that maximizes p(words,tags) given by the HMM.
        4. Compute and print the accuracy on the test data and other metrics or statistics
        Input: Test data, viterbi matrix, backtracking matrix
        Output: Predictions, Test stats (accuracy, known word accuracy, novel word accuracy)
    """

    test_words, groundtruth_tags = read_file(test_data)
    predicted_tags = ['###']
    

    #number of known words, number of novel words
    #prevent division by zero
    num_known , num_novel = 1e-15, 1e-15
    #correctly predicted novel and known tags
    correct_known, correct_novel = 0,0

    prev_tag = predicted_tags[0]

    #Compute the tag sequence that maximises p(words,tags) given by the HMM.
    for i in range(len(test_words)-1, 0, -1):
        # excluding the sentence boundary markers
        if test_words[i] != '###':
            if test_words[i] in word_counts:
                #token appear in the training data
                num_known += 1
                if predicted_tags[-1] == groundtruth_tags[i]:
                    correct_known += 1
            else:
                #token does not appear in the training data
                num_novel += 1
                if predicted_tags[-1] == groundtruth_tags[i]:
                    correct_novel  += 1

        predicted =  p[(i, prev_tag)]
        prev_tag = predicted
        predicted_tags.append(predicted)


    #predicted tags are reversed, reverse the list
    proper_predicted_tags = predicted_tags[::-1]

    #Evaluation
    #accuracy percentage of test tokens that received the corret tag, excluding the sentence boundary markers ###.
    acc = float(correct_known+correct_novel) *100/(num_known+ num_novel)

    #known word accuracy consider only tokens that appear in the training data.
    kwa = float(correct_known) *100/(num_known)
    #novel word accuracy consider only tokens that do not appear in the training data, in which case the model can only use the context to predict the tag.
    nwa = float(correct_novel) *100/(num_novel)
    print('Accuracy : {0}, KWA : {1}, NWA :{2} \n'.format(acc, kwa, nwa))
    return test_words, proper_predicted_tags, acc, kwa, nwa




def train_and_test(entrain,endev, entest):
    # The function should give an output file output.txt which is of the same format as (word/tag)
    # Read the training data and store the word-tag and tag-tag counts.

    transition_matrix, emission_matrix, pi, p = HMM(entrain, entest)
    test_words, proper_predicted_tags, acc, kwa, nwa = predictions(entest, pi, p)

    with open('output.txt', 'w') as f:
        for i in range(len(proper_predicted_tags)):
            s = str(test_words[i])+ "\\" + str (proper_predicted_tags[i])
            f.write('{}\n'.format(s))

    f.close()




if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='train dataset name')
    parser.add_argument('--test', help='test dataset name')
    args = parser.parse_args()

    train_and_test(args.train,None,args.test)



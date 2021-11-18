import numpy as np 
import argparse 
import math 

#define global matrices

# Keep the unigram counts of the tags and words for calculating matrices
word_counts, tag_counts = {}, {} 
#Word-tag and tag-tag counts
word_tag, tag_tag = {}, {}
#word-tag dictionary. Key: Word, Value: list of tags
WT_dict = {}
#Total number of tokens
N = 0
#Number of words in Vocab
V = 0
#Smoothning counts, all words given the tag and vice versa
smooth_tag_tag = {}
smooth_tag_word = {}
#alpha adder to prevent NaNs and zeros
alpha_adder = 1e-50

#SET EITHER BACKOFF OR INTERPOLATION TO BE TRUE. PREFERENCE TO BACKOFF

#decides whether to use the smoothened + backoff  probabilities (set to false for the vanilla HMM)
use_backoff = True
#decides whether to use interpolated transisition probabilities (set to false for the vanilla HMM)
use_interpolation = True
#decides whether to use feature extractors for tags (set to false for the vanilla HMM)
use_feature_extractor = True
#decides whether to print the known and unknown acc (used for debugging)
printVerbose = True


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


def forward_counts(filename):
    """ 
        1. Read the training data, and store the word-tag and tag-tag counts
        Input: Filename to be read from
        Output: Update global count variables
    """

    words, tags = read_file(filename)

    #update number of tokens
    N = len(words) - 1 #DONT take EOS

    #Account for the ### tokens
    #unigram counts of words
    word_counts[words[0]] = 1
    #unigram counts of tags
    tag_counts[tags[0]] = 1

    #word-tag dictionary. Key: Word, Value: list of tags
    WT_dict[words[0]] = [tags[0]]
    #Deal with OOV
    WT_dict['OOV'] = []

    #Word-Tag counts
    word_tag[(words[0],tags[0])] = 1

    smooth_tag_word[tags[0]] = 1

    for i in range(1, len(tags)):

        #Deal with OOV tags
        # For unseen words, all tags are allowed
        if tags[i] not in WT_dict['OOV']:    
            WT_dict['OOV'].append(tags[i])

        #Increase the unigram counts and the tag-tag occurrences
        word_counts[words[i]] = word_counts.get(words[i],0) + 1
        tag_counts[tags[i]] = tag_counts.get(tags[i],0) + 1
        tag_tag[(tags[i-1], tags[i])] =  tag_tag.get((tags[i-1], tags[i]), 0) + 1

        #Smooth tag-tags
        if tag_tag[(tags[i-1], tags[i])] == 1:
            smooth_tag_tag[tags[i-1]] = smooth_tag_tag.get(tags[i-1],0) + 1
        elif tag_tag[(tags[i-1], tags[i])] == 2:
            smooth_tag_tag[tags[i-1]] -= 1

        #For word-tag we may need to add into the dictionary as well.
        #If the (word-tag) pair has not been seen before, check if the word is in the dictionary. 
        #If not, add the tag. If it is in the dictionary, append the tag to the list of possible tags
        #You can derive a tag dictionary from the training set that records the set of allowed tags for each word, and only consider allowed tags for each word during viterbi decoding
        word_tag[(words[i], tags[i])] =  word_tag.get((words[i], tags[i]), 0) + 1
        #Smooth word-tags
        if word_tag[(words[i], tags[i])] == 1:
            if words[i] in WT_dict:
                WT_dict[words[i]].append(tags[i])
            else:
                WT_dict[words[i]]= [tags[i]]

            smooth_tag_word[tags[i]] = smooth_tag_word.get(tags[i],0) + 1
        elif word_tag[(words[i], tags[i])] == 2:
            smooth_tag_word[tags[i]] -= 1

    #update vocab len
    V = len(WT_dict.keys())

    return N,V

# Make sure to do all computation in the log space.
def transition_prob(tag1, tag2, backoff, interpolation,  N, V):
    """
        Calculate the transition probability from tag1 -> tag2 
        P(tag2|tag1) = count(tag1, tag2)/ count(tag1)

        with Laplacian smoothening we get
        P(tag2|tag1) = (count(tag1, tag2) + lambda)/ (count(tag1) + lambda*V)

        Try improving accuracy by using add1 smoothning + backoff
    """
    if backoff: 
        if interpolation:
            #backoff AND interpolation? #gives acc of 93.5043009888 for tuned lambda = 0.72 (higher autograder), 93.5444971461 at lambda = 0.9
            count_ = float(tag_counts.get(tag2, 0) + 1)/(N + V)
            alpha = smooth_tag_tag[tag1] + alpha_adder
            count_tag2 = float(tag_counts.get(tag2, 0) + 1)/(N + V)
            prob_tag1_tag2 = float(tag_tag.get((tag1, tag2),0) + alpha*count_)/(tag_counts[tag1]+ alpha)
            lamba = 0.72
            prob = lamba*(float(prob_tag1_tag2)) + (1-lamba)*count_tag2

        else:

            count_ = float(tag_counts.get(tag2, 0) + 1)/(N + V)
            alpha = smooth_tag_tag[tag1] + alpha_adder
            prob = float(tag_tag.get((tag1, tag2),0) + alpha*count_)/(tag_counts[tag1]+ alpha)

        

    elif interpolation:

        prob_tag1_tag2 = float(tag_tag.get((tag1, tag2),0)+1)/(tag_counts[tag1] + V)
        count_tag2 = float(tag_counts.get(tag2, 0) + 1)/(N + V)

        lamba = 0.82
        prob = lamba*(float(prob_tag1_tag2)) + (1-lamba)*count_tag2

    else:
       
        prob = float(tag_tag.get((tag1, tag2),0)+1)/(tag_counts[tag1] + V)

        
    return math.log(prob)

def emission_prob(tag, word, backoff, N, V):
    """
        Calculate the emission probability from tag -> word
        P(word|tag)  = count(word, tag)/ count(tag)

        with Laplacian smoothening we get
        P(word|tag) = (count(word, tag) + lambda)/ (count(tag) + lambda*V)

        Try improving accuracy by using add1 smoothning + backoff
    """
    if backoff: 
        count_ = float(word_counts.get(word, 0) + 1)/(N + V)
        alpha = smooth_tag_word[tag] + alpha_adder
        prob = float(word_tag.get((word, tag),0) + alpha*count_)/(tag_counts[tag]+ alpha)

    else:  
       
        prob = float(word_tag.get((word, tag),0)+1)/(tag_counts[tag] + V)

 

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
    N, V = forward_counts(train_data)

    #initialize pi and v
    #Unseen word?
    for tag in WT_dict.get(test_words[0], WT_dict['OOV']):
        # probability =  emission + transisition (calculations in log space)
        pi[(0, tag)] =  emission_prob(tag, test_words[0], use_backoff, N, V) + transition_prob('###', tag, use_backoff,use_interpolation, N, V)
        # all first words backpoint to the start tag
        p[(0, tag)] = '###' #Udit's OH: Assume it starts with ###

    #fill in the tables (follow method by Prof in class - check for every tag preceding all possible tags at currrent pos i)
    #for each word at current position i
    for i in range(1, len(test_words)):
        #for each posible tag ti at pos i
        for ti in WT_dict.get(test_words[i], WT_dict['OOV']):
            #for each tag tj before ti
            for tj in WT_dict.get(test_words[i-1], WT_dict['OOV']):
                #check if emission and transition probabilities exist
                if (tj, ti) not in transition_matrix:
                    transition_matrix[(tj, ti)] = transition_prob(tj, ti, use_backoff, use_interpolation, N, V)
                if (ti, test_words[i]) not in emission_matrix:
                    emission_matrix[(ti, test_words[i])] = emission_prob(ti, test_words[i], use_backoff, N, V)
                
                mu = emission_matrix[(ti, test_words[i])] + transition_matrix[(tj, ti)] + pi[(i-1, tj)]

                #find argmax
                if mu > pi.get((i, ti), float('-inf')):
                    pi[(i, ti)] = mu
                    p[(i, ti)] = tj
        
            #feature extraction: after analysing the incorrect tags, I found that two main types of words were being misclassified:
            # 1. when the word ends with $ it should be tagged $, 2. numbers should be C 
            if use_feature_extractor:
                word_ends_with_dollar = 1 if '$' in test_words[i-1] else 0
                word_is_number = 1 if test_words[i-1].replace('.','').replace('-','').isdigit() else 0

                if word_ends_with_dollar:
                    p[(i, ti)] = '$'
                else:
                    if word_is_number:
                        p[(i, ti)] = 'C'
                    


    return transition_matrix, emission_matrix, pi, p
    

def predictions(test_data, pi, p, printVerbose):
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
    num_known , num_novel = 0,0
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
    if num_known==0:
        if printVerbose:
            print('No Known words in test data \n')
        kwa = 0.0
    else:
        kwa = float(correct_known) *100/(num_known)

    #novel word accuracy consider only tokens that do not appear in the training data, in which case the model can only use the context to predict the tag.
    if num_novel==0:
        if printVerbose:
            print('No Novel words in test data \n')
        nwa = 0.0
    else:
        nwa = float(correct_novel) *100/(num_novel)

    if printVerbose:
        print('Accuracy : {0}, KWA : {1}, NWA :{2} \n'.format(acc, kwa, nwa))
    return test_words, proper_predicted_tags, acc, kwa, nwa


def eval(gold, pred):
    """
        My implementation of eval.py (correct accuracy to be a float*100). Used to check whether the output format is correct in train_and_test.
    """
    readline = lambda line: line.strip().split('/')
    words = []
    groundtruth_tags = []
    predicted_tags = []
    with open(gold) as fgold, open(pred) as fpred:
        for g, p in zip(fgold, fpred):
            gw, gt = readline(g)
            pw, pt = readline(p)
            if gw == '###':
                continue
            words.append(gw)
            predicted_tags.append(pt)
            groundtruth_tags.append(gt)
    
    acc = float(sum([pt == gt for gt, pt in zip(groundtruth_tags, predicted_tags)])) / len(predicted_tags)
    print('accuracy={}'.format(acc*100))
    return acc



def train_and_test(entrain,endev, entest):
    """
    Input: Training, Dev and Test data set paths
    Output: The function should give an output file output.txt which is of the same format as (word/tag)
    """

    transition_matrix, emission_matrix, pi, p = HMM(entrain, entest)
    test_words, proper_predicted_tags, acc, kwa, nwa = predictions(entest, pi, p, printVerbose)

    with open('output.txt', 'w') as f:
        for i in range(len(proper_predicted_tags)):
            s = str(test_words[i])+ "/" + str (proper_predicted_tags[i])
            f.write('{}\n'.format(s))

    f.close()

    #check if output is correctly formatted
    #acc = eval(entest, 'output.txt')
    

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='train dataset name')
    parser.add_argument('--test', help='test dataset name')
    parser.add_argument('--backoff', help='decides whether to use the smoothened + backoff probabilities (set to false for the vanilla HMM)')
    parser.add_argument('--interpolation', help='decides whether to use interpolated transisition probabilities (set to false for the vanilla HMM)')
    parser.add_argument('--feature_extractor', help='decides whether to use feature extractors for tags (set to false for the vanilla HMM)')
    parser.add_argument('--verbose', help='#decides whether to print the known and unknown acc (used for debugging)')
    args = parser.parse_args()

    #decides whether to use the smoothened + backoff  probabilities (set to false for the vanilla HMM)
    use_backoff = args.backoff
    #decides whether to use interpolated transisition probabilities (set to false for the vanilla HMM)
    use_interpolation = args.interpolation
    #decides whether to use feature extractors for tags (set to false for the vanilla HMM)
    use_feature_extractor = args.feature_extractor
    #decides whether to print the known and unknown acc (used for debugging)
    printVerbose = args.verbose

    train_and_test(args.train,None,args.test)




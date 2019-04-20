from collections import Counter, defaultdict

def rearrange_data(sequences):
    x = []
    y = []
    w = set()
    t = set()
    for sequence in sequences:
        sequence_x = []
        sequence_y = []
        for word, tag in sequence:
            sequence_x.append(word)
            sequence_y.append(tag)
            w.add(word)
            t.add(tag)
        x.append(sequence_x)
        y.append(sequence_y)
    return x, y, w, t

def replace_unknown(sequence, vocabulary):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in vocabulary else 'nan' for w in sequence]

def simplify_decoding(X, model, vocabulary):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X, vocabulary))
    return [state[1].name for state in state_path[1:-1]]  # do not show the start/end state predictions

def accuracy(X, Y, model, vocabulary):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.
    
    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.
    
    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [(), (), ...]
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model, vocabulary)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions

def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    counts = defaultdict(Counter)
    for tags, words in zip(sequences_A, sequences_B):
        for tag, word in zip(tags, words):
            counts[tag][word] += 1
    return counts

def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.
    
    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    counts = Counter()
    for sequence in sequences:
        for tag in sequence:
            counts[tag] += 1
    return counts

def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
    counts = Counter()
    for sequence in sequences:
        for tag1, tag2 in zip(sequence[:-1], sequence[1:]):
            counts[(tag1, tag2)] += 1
    return counts

def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.
    
    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    counts = Counter()
    for sequence in sequences:
        counts[sequence[0]] += 1
    return counts

def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.
    
    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """
    counts = Counter()
    for sequence in sequences:
        counts[sequence[-1]] += 1
    return counts
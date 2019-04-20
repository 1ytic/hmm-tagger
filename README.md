## Introduction

In the course [Natural Language Processing Nanodegree Program](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892) there is a project called "Part of Speech Tagging", in which you need to create a part of speech tagging system using hidden Markov models. The last step of the project challenges students to create more advanced HMMs.

In this repository, I'll try to cover solutions for the Step 4 of the project "Part of Speech Tagging". In particular, I'll use the [Pomegranate](https://github.com/jmschrei/pomegranate) library to build a hidden Markov model for part of speech tagging with the Brown corpus with the full NLTK tagset.

Refer to [Chapter 5](http://www.nltk.org/book/ch05.html) of the NLTK book for more information about the corpus and the available tagsets.

## Improving model performance

There are additional enhancements that can be incorporated into your tagger that improve performance on larger tagsets where the data sparsity problem is more significant. The data sparsity problem arises because the same amount of data split over more tags means there will be fewer samples in each tag, and there will be more missing data tags that have zero occurrences in the data.

- [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) (pseudocounts)
    Laplace smoothing is a technique where you add a small, non-zero value to all observed counts to offset for unobserved values.

- Backoff Smoothing
    Another smoothing technique is to interpolate between n-grams for missing data. This method is more effective than Laplace smoothing at combatting the data sparsity problem. Refer to chapters 3 and 8 of the [Speech & Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book for more information.

- Extending to Trigrams
    HMM taggers have achieved better than 96% accuracy with the full Penn treebank tagset using an architecture described in [this](http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf) paper. Altering your HMM to achieve the same performance would require implementing deleted interpolation (described in the paper), incorporating trigram probabilities in your frequency tables, and re-implementing the Viterbi algorithm to consider three consecutive states instead of two.
    
### Baseline

[baseline.ipynb](https://github.com/1ytic/hmm-tagger/baseline.ipynb)

The baseline hidden Markov model have been able to achieve >92% tag accuracy with larger tagsets on realistic text corpora.
    
### Laplace Smoothing

[laplace-smoothing.ipynb](https://github.com/1ytic/hmm-tagger/laplace-smoothing.ipynb)

In this notebook, I smoothed only transition probability from tag to tag, and added unknown transitions, which tags exists in the training data. This technique improve the baseline model for 1% in absolute and achieve >93% tag accuracy.

### TODO:

- Backoff Smoothing

- Extending to Trigrams
# Gensim topic modeling with Hierarchical Dirichlet Process
## (on Twitter)

## Purpose and Background

This script extracts key terms from related documents and examines them in the context of social media hashtags.

A challenge in analyzing the data in this preliminary use case is the very small corpus size. While I also used some other models incorporating transfer learning and document-length independent graph-based algorithms (e.g., variations on LexRank and TextRank, other summarization methods for keyword extraction, and comparisons with pretrained vector models), I decided to start with a model using HDP because:

1. It accounts for the interrelatedness among the documents by permitting shared vocabulary among topics;
2. It is extensible, both in that the model can be updated online and in that the mathematics allow for extension to multiple corpora, where it may be of interest to discover latent topics that are shared among document clusters;
3. Unlike LDA, it "learns" the number of topics through training, meaning it is well-suited to unsupervised learning in which the topic distribution is not known a priori.

Edge cases remain, improvements will be made, and I am extending the corpus and incorporating TextRank and other graph-based models. 

Still, this model provides a reasonable balance between stability of topic cohesion and the desirable serendipity that arises from the stochastic processes that provide its theoretical underpinning.

## How to use

These scripts were run from the command line in a folder containing the subdirectory of files to be processed. 

'''
python -m preprocessing

python -m hdp_model
'''


## What you see

The output consists of (1) an HTML file containing a table of terms/hashtags, the documents in which they appear, and the sentences in which they appear; (2) a PDF of same; and (3) a dictionary and corpus saved to disk (N.B.: The filetype and location of storage could be easily modified for a larger/extended corpus).

## The #twittersphere

What if these were real hashtags? Using authentication, the model-generated terms were fed into the Twitter API to retrieve 5 "top" tweets (by relevance and timeliness) from Twitter for each word, and roughly analyzed for sentiment polarity with VADER. The results are stored in csv files for now. 

## Next steps

1. Include additional algorithms and transfer learning for better performance
2. Use the corpus of presidential campaign speeches
3. Analyze ProPublica's [Politwhoops](https://projects.propublica.org/politwoops/about) database

Thanks for reading!
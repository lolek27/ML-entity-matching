# Entity matching

Entity matching is the problem of identifying which records refer to the same real-world entity.
This repository contains exercise on entity matching problem, implementing some of the successful techniques from my own
research on publicly available datasets.

 
For the exercise I've picked two small datasets for product matching task:

1. **Abt - Buy dataset** - around 7k of electronic devices product matches between Abt and Buy internet stores,
where positively matched pairs constitute about 15% of all data points they've gathered
2. **Amazon - Google dataset** - about 8k dataset of also electronic/software product matches between Google and Amazon store
 with similar ratio of positive matches to all pairs to Abt-Buy

 
Based on [this](https://paperswithcode.com/sota/entity-resolution-on-amazon-google) benchmark the best F1 score for Amazon-Google 
is close to 80% and is reached through techniques described in [Supervised Contrastive Learning for Product Matching](https://paperswithcode.com/paper/supervised-contrastive-learning-for-product/review/?hl=46830)
paper from 2022. The same paper mentions an F1 score of 94.29% for Abt-Buy.

Assuming it's the F1 score for the positive class, with techniques I've helped developed I could get to:
1.   **96.68%** F1 score for Abt-Buy (compare to 94.68% from paper above)
2.   **94.83%** F1 score for Amazon-Google (compare to 80% from paper above)

What I've tried in short is:
*   [Sentence Transfomers](https://www.sbert.net/) embeddings - SOTA sentence, text and image embeddings
*   [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - efficient vector similarity search
*   Data augmentation
 

### Results:

*  With pretrained sentence transformers embeddings + scaNN I could get to about **70%** F1 score for positive class.
  The embeddings where not sensitive enough to subtle differences in product models names, for instance like *LCSX20* and *LCSMX100*.
*  Fine tuning the embeddings with the use of *Multiple Negative Ranking Loss* improved the scores by **20% to 30%**.
  MNR loss is a great choice to consider when we do not possess a lot of positive data points (in our case - positively matched pairs constitute only about 15% of all data points)
MNRL assemebles a set of negative pairs for each data point by assuming that every other item is a negative match - hence greatly enhancing on trainig set size.
*  Additional augmentation of data consisting of several techniques (swapping, inserting, deleting, cropping words and
  characters, also using similar words instead) further improved the results by **~2%** for both Abt-Buy and Amazon-Google datasets.

The exercise isn't exhaustive (further tweaking some hyper parameters could probably nudge the scores even higher),
but it proves how pawerful and universal the applied search and match techniques are with just a little effort.


Recall charts for top 10 candidates after scaNN:

Abt-Buy:

![unnamed](https://github.com/lolek27/ML-entity_matching/assets/12550403/85453744-6ed4-4d0f-b97b-340b27a8568c)

Amazon-Google:

![unnamed](https://github.com/lolek27/ML-entity_matching/assets/12550403/04d3833d-2161-4566-861f-d903f6a41885)

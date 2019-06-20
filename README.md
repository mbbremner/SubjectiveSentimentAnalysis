# SubjectiveSentimentAnalysis
This is a sentiment analysis project using recurrent neural networks. Subjective standards of data 
separation are explored. This work is motivated by a desire to expand my abilitie wrt NLP. 
In this work, the point of  view of a single public figure stands in as a pseudo-subjective definition 
of sentiment. Text comments are separated with an LSTM network based on their similarity to text comments
made by the public figure of concern. To make things interesting, an archive of Tweets from a controversial
public figure serves as a pseudo-subjective ground truth. A large database of YouTube comments are separated in
an abstract feature space based on their similarity to tweets made by Donald Trump.

There are some aspects of this project which are ill-conceived. For instance, a clearly problematic choice
was made to compare text comments across domains (YouTube comments vs. Tweets). Some difficulties
regarding cross domain textual analysis are discussed within the report.

The final product is a fu
The initial results are banal and uninteresting. 

Additionally, I've applied a similar analysis of my own personal history of YouTube comments.

Although this is a complete pipeline, the work is ongoing and motivated beyond its current form.

Nobody will ever give a specific definition of something bad that happened that point of view would
agree is 

This project is complete text analysis pipeline for sentiment analysis. The motivation is to 
explore the utility of subjective definitions of sentiment in text sentiment analysis.

The function of the LSTM network is to separate text sequences in an abstract features space.
Credit also to Justin Brown whose contributions are worth mentioning.

# This Project Includes Scripts for ...

1. Data Acquisition with YouTube API v3
2. Data Pre-processing
3. Data Partitioning
4. Training an LSTM with mxnet
5. Feature extraction
6. Principle Component Analysis & Visualization


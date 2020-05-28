# Information-Retrieval-Demo
Using TF-IDF and Cosine Similarity

A quick demo on Information Retrieval that uses the TF-IDF model and Cosine Similarity to rank documents. The program takes a set of documents (in this case the ClueWeb09 corpus) and preprocesses them (remove punctuation characters, removes markup, tokenize,  removed stopwords, stem words). The program then calculates TF-IDF scores and ranks the documents using cosine similarity based on a list of queries in the 'topics.txt' file. The program will generate a results file called 'tfidf_results.test' to be used with the TREC eval program.

To run the program, simply run the 'tfresults.py' code with the corpus file path set to the appropriate location. NOTE: the program only works with a small sample size (tested with only 200 documents).

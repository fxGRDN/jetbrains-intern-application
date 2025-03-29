# jetbrains-intern-application
Task for jetbrain internship

## DAY 1
Setup

## DAY 2

### How to perform evaluation?

FixedTokenChunker chunks text based of fixed lenght of **tokens subset**.
We want to find how diffrent chunking methods make finding relevant information easier or harder.

Given a question we already have relevant excerpts. What we want to do is find starting and ending indexes of each chunk in the text and compare it in some way to indexes of excerpts.

By measuring the mean of IoU score we get to know how chunker performs.

### Evaluation algorith

To evaluate metrics such as IoU, precision, reall or F1 we need two components:
1. Intersection of excerpts and chunks,
2. Size of both excerpts and chunks

For overlaping chunks we have to differenciate the excerpt range for every chunk.

## DAY 3


### Retrival evaluation pipeline

What we want to accomplish here? By using embedder, we want to find top-k relevant chunks based on cosine similiarity of chunks and query. Then we evaluate the quality of retrived chunks using previously defined metrics. If we are close to the relevant information then it means that chunker did it job pretty well.

## DAY 4

Cleaning up the code.
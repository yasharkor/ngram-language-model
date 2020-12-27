# N-gram language model

## Installation and Execution Instructions
- Clone the repository
- Using a terminal navigate to the "code" subdirectory (assignment-3-language-models-yasharkor/code as on github)
- No extra python libraries needed outside of standard library (re, sys, os, math, csv)
- From the terminal run the command ```python3 .\langid.py <train data path> <test data path> <output path> <mode>```
- The train data path should be a relative path to a directory containing the text files you will use to build language models
- The test data path should be a relative path to a directory containing the text files you would like to evaluate
- The output path is where the csv files will be created after evaluating your test data
- The accepted modes are as follows:
  - "--unsmoothed" for generating language models without smoothing
  - "--laplace" for generating language models with add one laplace smoothing
  - "--interpolation" for generating language models with deleted interpolation
- Additionally there are two other modes:
  - "--bestUnsmoothed" for finding the best N for your test data with unsmoothed language models (used for tuning n with development data)
  - "--bestLaplace" for finding the best N for your test data with add one Laplace language models (used for tuning n with development data)

## Resources
  - Speech and Language Processing Book by Daniel Jurafsky is used.
  - Python documentation for standard libraries
  - https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list/3989029


## Output files
Running this program in any of the standard 3 modes outputs a csv file of the form: ```Test File, Best Training Match, Perplexity, N ```

## Dependency
The assignment's training data can be found in [data/train](data/train) and the development data can be found in [data/dev](data/dev).


## Introduction

Language models have various interesting applications. In this project, we will explore an application of language models to automatically identify the language of a text. The frequency distribution of character n-grams varies between languages. Therefore, a character language model will typically assign higher probabilities (and so, lower perplexities) to text in the same language as the text it was trained on. For example, a character language model trained on English will, on average, assign a lower perplexity to an English sentence than a German sentence. In this project, we will use this property of language models for language identification.
 ## Task

The training set includes 55 files with names of the form 'udhr-.*.txt.tra'. Each file contains part of the Universal Declaration of Human Rights (UDHR) in various languages. The task is to use the texts to induce character n-gram language models for the different languages. 

The development set contains a list of files with names of the form 'udhr-.*.txt.dev'. These files contain another, smaller, part of the UDHR from the same set of languages. The task is to use these files to test language models during development. 

Once your language models are constructed based on your training set, they will be applied to the files in the development set to produce perplexities. A language model with the lowest perplexity indicates the file contains the same language. 

The program allow three different types of character language models: 
- Unsmoothed model 
- Smoothed with add-one (Laplace) smoothing 
- Smoothed with linear interpolation smoothing

For interpolation, the lambda coefficients is determined using the deleted interpolation algorithm, as described in the J&M textbook ([chapter 8](https://web.stanford.edu/~jurafsky/slp3/3.pdf), pages 15-16, 3rd edition, chapter 3, page 12 for OOV).

For the three types of model, we tune the parameter n.  The parameter n should control the size of the context used by the model and it may be different for each type of model but remains the same across all languages within the model. Thus, we have exactly three values of n to tune.

## Useful Links
[Chapter 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) of J+M introduces N-gram Language Models. [Chapter 8 (p.15-16)](https://web.stanford.edu/~jurafsky/slp3/8.pdf) introduces the deleted interpolation algorithm.

## Acknoeldgements


This is a solution to Assignment 3 for CMPUT 497 - Intro to NLP at the University of Alberta, created during the Fall 2020 semester. ##  Yashar Kor: yashar@ualberta.ca, Thomas Maurer: tmaurer@ualberta.ca
import os
import sys
import re
import csv
from math import log


# Returns the log base 2 of a given float
def log_base2(probability):
    if probability == 0.0:
        return float("-inf")
    return log(probability, 2)

# Returns the paths to all files in the directory given
def getAllFilePaths(dirPath):
    return ['%s/%s' % (dirPath, f) for f in os.listdir(dirPath) if os.path.isfile('%s/%s' % (dirPath, f))]

# Returns a dict of ngram counts
# Dict changes form depending on if n == 1 or not
# n = 1 -> dict = {unigram: count}
# n > 1 -> dict = {ngram_up_to_last_letter: {last_letter: count}}
def getNgramCounts(n, filePath):
    counts = {}

    with open(filePath) as f:
        # get the text and replace new lines with spaces
        text = f.read()
        text = re.sub('\n', ' ', text).strip()

        # get the possible characters
        vocab = set(list(text))

        if n == 1:
            # get each character in the text
            for i in range(len(text)):
                ngramPartOne = text[i]

                # check whether that character has been seen before or not
                if ngramPartOne in counts.keys():
                    counts[ngramPartOne] += 1 # increment count
                else:
                    counts[ngramPartOne] = 1 # initialize count
        # n > 1
        else:
            # get each starting character of an ngram
            for i in range(len(text) - (n) + 1):
                ngramPartOne = ''.join(text[i: (i + n - 1)]) # the part up to the last character
                ngramPartTwo = text[(i + n - 1)] # the last character
                
                # nested check does the same as the n == 1 but for both dimensions
                if ngramPartOne in counts.keys():
                    if ngramPartTwo in counts[ngramPartOne]:
                        counts[ngramPartOne][ngramPartTwo] += 1
                    else:
                        counts[ngramPartOne][ngramPartTwo] = 1
                else:
                    counts[ngramPartOne] = {ngramPartTwo: 1}

    return counts, vocab


# Calculates and returns the unsmoothed perplexity of a file
# with respect to the counts of a language model
def perplexity_noSmoothing(n, filePath, counts):
    probabilityList = list()
    with open(filePath) as f:
        # get the text and replace new lines with spaces
        text = f.read()
        text = re.sub('\n', ' ', text).strip()

        # all first characters of the ngrams
        for i in range(len(text) - (n)+1):
            # unigrams
            if n == 1:
                ngramPartOne = text[i]
                if ngramPartOne in counts.keys():
                    # get the count of the unigram divided by the total unigrams
                    partTwoCount = counts[ngramPartOne]
                    partOneCount = sum(counts.values())
                    probab = (partTwoCount)/(partOneCount)
                else: # never seen the unigram before
                    probab = 0

            else:
                ngramPartOne = ''.join(text[i: (i + n - 1)])  # the part up to the last character
                ngramPartTwo = text[(i + n-1)] # the last character

                if ngramPartOne not in counts.keys(): # never seen the ngram before
                    probab = 0
                elif ngramPartTwo not in counts[ngramPartOne].keys(): # never seen the ngram before
                    probab = 0
                else:
                    # get the count of the ngram divided by the total ngrmas
                    partTwoCount = counts[ngramPartOne][ngramPartTwo]
                    partOneCount = sum(counts[ngramPartOne].values())
                    probab = (partTwoCount)/(partOneCount)
            # store the probability
            probabilityList.append(log_base2(probab))

        # -1 * mean of a probabilityList
        # PP(W) = 2^-l where l = (1/N)(log(P(w)))
        entropy = -1*(sum(probabilityList)/len(probabilityList))
        perplexity = pow(2, entropy)
        return perplexity


# Calculates the add-one Laplace perplexity of a file
# with respect to a given language model counts and vocab
def perplexity_Laplace(n, filePath, counts, vocab):
    # initialize
    probabilityList = list()
    vocab_size = len(vocab)
    with open(filePath) as f:
        # get the text and replace new lines with spaces
        text = f.read()
        text = re.sub('\n', ' ', text).strip()

        # This section is all the same as the unsmoothed perplexity
        # except that it remembers the counts for now instead of
        # calculating the probability
        for i in range(len(text) - (n)+1):
            if n == 1:
                ngramPartOne = text[i]
                if ngramPartOne in counts.keys():
                    partTwoCount = counts[ngramPartOne]
                else:
                    partTwoCount = 0
                partOneCount = sum(counts.values())

            else:
                ngramPartOne = ''.join(text[i: (i + n - 1)])
                ngramPartTwo = text[(i + n-1)]

                if ngramPartOne not in counts.keys():
                    partOneCount = 0
                    partTwoCount = 0
                elif ngramPartTwo not in counts[ngramPartOne].keys():
                    partOneCount = sum(counts[ngramPartOne].values())
                    partTwoCount = 0
                else:
                    partTwoCount = counts[ngramPartOne][ngramPartTwo]
                    partOneCount = sum(counts[ngramPartOne].values())

            partTwoCount += 1 # add one to the ngram count
            partOneCount += vocab_size # add the vocab_size to the denominator

            # calculate the probability and save it
            probab = (partTwoCount)/(partOneCount)
            probabilityList.append(log_base2(probab))

        # -1 * mean of a probabilityList
        # PP(W) = 2^-l where l = (1/N)(log(P(w)))
        entropy = -1*(sum(probabilityList)/len(probabilityList))
        perplexity = pow(2, entropy)
        return perplexity


# Calculates the interpolation perplexity for a file with
# respect to a given language model counts, and some lambda
# weights for each ngram
def perplexity_Interpolation(n, lambdas, filePath, counts):
    probabilityList = list()

    with open(filePath) as f:
        # get the text and replace new lines with spaces
        text = f.read()
        text = re.sub('\n', ' ', text).strip()

        # iterate through each max-length ngram starting letter
        for i in range(len(text) - n+1):
            probab = 0 # start probability at 0
            for nLength in range(n): # for each sub n gram
                partialProbab = 0 # partial probability starts at 0
                if nLength + 1 == 1: # unigram case
                    ngramPartOne = text[i] # get the unigram

                    # if not unknown calculate a partial probability for
                    # this sub ngram
                    if ngramPartOne in counts[nLength + 1].keys(): 
                        partOneCount = counts[nLength + 1][ngramPartOne]
                        partTwoCount = sum(counts[nLength+1].values())
                        partialProbab = (partOneCount)/(partTwoCount)
                else: # not a unigram
                    # get the keys for the count
                    ngramPartOne = ''.join(text[i: (i + nLength)])
                    ngramPartTwo = text[(i + nLength)]

                    if ngramPartOne not in counts[nLength+1].keys(): #unknown
                        partialProbab += 0
                    elif ngramPartTwo not in counts[nLength+1][ngramPartOne].keys():#unknown
                        partialProbab += 0
                    else: #known ngram so add the probability
                        partOneCount = counts[nLength + 1][ngramPartOne][ngramPartTwo]
                        partTwoCount = sum(counts[nLength+1][ngramPartOne].values())
                        partialProbab = (partOneCount)/(partTwoCount)
                # multiply partial probability by the weight and add to the probability
                probab += lambdas[nLength] * partialProbab

            # save the probability
            probabilityList.append(log_base2(probab))

        # -1 * mean of a probabilityList
        # PP(W) = 2^-l where l = (1/N)(log(P(w)))
        entropy = -1*(sum(probabilityList)/len(probabilityList))
        perplexity = pow(2, entropy)
        return perplexity

# Outputs a csv of results based on the mode
def outputCSV(rows, mode, outputDir):
    # select the title
    if mode == '--unsmoothed':
        title = 'results_dev_unsmoothed.csv'
    elif mode == '--laplace':
        title = 'results_dev_add-one.csv'
    elif mode == '--interpolation':
        title = 'results_dev_interpolation.csv'
    elif mode == '--bestUnsmoothed':
        return
    elif mode == '--bestLaplace':
        return
    elif mode == '--bestInterpolation':
        return
    else:
        print('incorrect mode')
        exit()
    # write to file
    with open(outputDir + '/' + title, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(rows)

# Main loop for unsmoothed evaluation
def unsmoothed(dirPath, dirPath2, n, rows):
    # Get the training data counts
    langModels = {}
    for f in getAllFilePaths(dirPath):
        counts, _ = getNgramCounts(n, f)
        langModels[f.split('/')[-1]] = counts

    # Check each test file
    for de in getAllFilePaths(dirPath2):
        bestFile = 'no matches'
        minPerplexity = float('inf')
        dev_file_name = de.split('/')[-1]
        
        # Calculate the perplexity with respect to each training file
        for tr in getAllFilePaths(dirPath):
            train_file_name = tr.split('/')[-1]
            p = perplexity_noSmoothing(n, de, langModels[train_file_name])

            # save the training file if it as the lowest perplexity
            if p < minPerplexity:
                bestFile = train_file_name
                minPerplexity = p
        # Save the best match as a row
        rows.append([bestFile, dev_file_name, minPerplexity, n])
    return rows

# Main loop for laplace evaluation
def laplace(dirPath, dirPath2, n, rows):
    # Get the training data counts
    langModels = {}
    langModelsVocab = {}
    for f in getAllFilePaths(dirPath):
        counts, vocab = getNgramCounts(n, f)
        langModels[f.split('/')[-1]] = counts
        langModelsVocab[f.split('/')[-1]] = vocab

    # Check each test file
    for de in getAllFilePaths(dirPath2):
        bestFile = 'no matches'
        minPerplexity = float('inf')
        dev_file_name = de.split('/')[-1]

        # Calculate the perplexity with respect to each training file
        for tr in getAllFilePaths(dirPath):
            train_file_name = tr.split('/')[-1]
            p = perplexity_Laplace(n,de,langModels[train_file_name],langModelsVocab[train_file_name])

            # save the training file if it as the lowest perplexity
            if p < minPerplexity:
                bestFile = train_file_name
                minPerplexity = p

        # Save the best match as a row
        rows.append([bestFile, dev_file_name, minPerplexity, n])
    return rows

# Main loop for interpolation evaluation
def interpolation(dirPath, dirPath2, n, rows):
    langModels = {}
    lambdas = {}

    # Get the training data counts
    for f in getAllFilePaths(dirPath):
        langModels[f.split('/')[-1]] = {}
        unifiedCounts = []

        # get a count for each sub ngram length
        for i in range(n):
            counts, _ = getNgramCounts(i + 1, f)
            langModels[f.split('/')[-1]][i+1] = counts

            # get the counts of ngrams as a 1D dictionary for ease of lambda calculation
            if i + 1 == 1:
                unifiedCounts.append(langModels[f.split('/')[-1]][i+1])
            else:
                unifiedCounts.append(getUnifiedCounts(langModels[f.split('/')[-1]][i+1]))
        
        # get the lambdas for each file based on the unified counts using deleted interpolation
        lambdas[f.split('/')[-1]] = deletedInterpolation(unifiedCounts)

    # Check each test file
    for de in getAllFilePaths(dirPath2):
        bestFile = 'no matches'
        minPerplexity = float('inf')
        dev_file_name = de.split('/')[-1]

        # Calculate the perplexity with respect to each training file
        for tr in getAllFilePaths(dirPath):
            train_file_name = tr.split('/')[-1]
            p = perplexity_Interpolation(n,lambdas[train_file_name],de,langModels[train_file_name])

            # save the training file if it as the lowest perplexity
            if p < minPerplexity:
                bestFile = train_file_name
                minPerplexity = p
        
        # Save the best match as a row
        rows.append([bestFile, dev_file_name, minPerplexity, n])
    return rows

# Calculates the best lambdas for a language model give
# a set of ngram counts
def deletedInterpolation(ngrams):
    # initialize lambdas
    lambdas = [0] * len(ngrams)

    # for each ngram of max length
    for g in ngrams[-1]:
        cases = [0]*len(ngrams) # intialize cases

        # for all sub ngrams
        for i in range(len(ngrams)):
            if i == 0: # unigram case
                lastLetter = g[-1] # the unigram we want to check
                counts = ngrams[i] # the counts of unigram
                numerator = counts[lastLetter] - 1 # the count of the unigram we are checking - 1
                denominator = sum(counts.values()) -1 # the total unigrams
                cases[i] = numerator / denominator if denominator != 0 else 0 # store the value
            elif i == 1: # bigram, unigram denominator
                secondLastLetter = g[-2] # the first letter of the bigram
                lastLetter = g[-1]  # the second letter of the bigram, and the unigram
                bigramCounts = ngrams[i] # the counts of all bigrams
                unigramCounts = ngrams[i-1] # the counts of all unigrams
                numerator = bigramCounts[secondLastLetter + lastLetter] - 1 # the count of the specific bigram - 1
                denominator = unigramCounts[secondLastLetter] - 1 # the count of the specific unigram - 1
                cases[i] = numerator / denominator if denominator != 0 else 0 # store the value
            else: # all others
                top = g[-(i+1):] # The full length ngram for some sub ngram
                bottom = g[-(i+1):-1] # the same ngram without the last letter (partial ngram)
                topCounts = ngrams[i] # the counts of the sub ngram length 
                bottomCounts = ngrams[i-1] # the counts of the partial ngrams
                numerator = topCounts[top] - 1 # the specific count of the sub ngram - 1
                denominator = bottomCounts[bottom] -1 # the specific count of the partial ngram - 1
                cases[i] = numerator / denominator if denominator != 0 else 0 # store the value
        
        # Find the max case
        maxValue = max(cases)
        # get the index of the max case
        idx = [i for i, j in enumerate(cases) if j == maxValue][0]
        # increment the lambda at that index by the count of the full length ngram
        lambdas[idx] += ngrams[-1][g]

    # normalize the lambdas
    lambdas = [l / sum(lambdas) for l in lambdas]
    return lambdas

# Flattens a 2d ngram array into a 1d ngram array
def getUnifiedCounts(counts):
    grams = {}
    for key in counts.keys():
        for key2 in counts[key]:
            grams[key+key2] = counts[key][key2]
    return grams

# Returns the accuracy of a results file
# given the rows of that results file
def getAccuracy(rows):
    total = 0
    correct = 0
    incorrect = []
    for i in range(1, len(rows)):
        row = rows[i]
        if row[0].split('.')[0] == row[1].split('.')[0]: # training language == test language
            correct += 1
        else:
            incorrect += [row]
        total += 1
    
    return correct / total

# A loop for finding the best N using accuracy of results against a
# test/validation/dev set
# dirpath = training data
# dirpath2 = test data
# mode = mode to use
def findBestN(dirPath, dirPath2, mode):
    best = 1
    bestAccuracy = 0
    # depending on the mode loop through n values of 1 to 19
    # and calculate the accuracy of the results. Then save the
    # value and return it
    if mode == '--bestUnsmoothed':
        for n in range(1, 20):
            rows = []
            rows = unsmoothed(dirPath, dirPath2, n, rows)
            accuracy = getAccuracy(rows)
            if accuracy > bestAccuracy:
                best = n
                bestAccuracy = accuracy
    elif mode == "--bestLaplace":
        for n in range(1, 20):
            rows = []
            rows = laplace(dirPath, dirPath2, n, rows)
            accuracy = getAccuracy(rows)
            if accuracy > bestAccuracy:
                best = n
                bestAccuracy = accuracy
    elif mode == "--bestInterpolation":
        for n in range(1, 20):
            rows = []
            rows = interpolation(dirPath, dirPath2, n, rows)
            accuracy = getAccuracy(rows)
            if accuracy > bestAccuracy:
                best = n
                bestAccuracy = accuracy
    return best, bestAccuracy

def main():
    # directory of Training files
    dirPath = sys.argv[1]
    # directory of Developement files
    dirPath2 = sys.argv[2]
    # directory of Output files
    dirPath3 = sys.argv[3]
    # smoothing mode
    mode = sys.argv[4]

    # Create header
    rows = [['Training_file', 'Testing_file', 'Perplexity' , 'N']]
    if mode == '--unsmoothed':
        n = 1 # calculated using best N function
        rows = unsmoothed(dirPath, dirPath2, n, rows)
    elif mode == "--laplace":
        n = 4 # calculated using best N function
        rows = laplace(dirPath, dirPath2, n, rows)
    elif mode == '--interpolation':
        n = 3 # calculated using best N function
        rows = interpolation(dirPath, dirPath2, n, rows)
    
    # extra modes for calculating and printing best N
    elif mode == '--bestUnsmoothed':
        print('Unsmoothed best N: %d, with accuracy: %.3f' %(findBestN(dirPath, dirPath2, mode)))
    elif mode == '--bestLaplace':
        print('Laplace best N: %d, with accuracy: %.3f' %(findBestN(dirPath, dirPath2, mode)))
    elif mode == '--bestInterpolation':
        print('Interpolation best N: %d, with accuracy: %.3f' %(findBestN(dirPath, dirPath2, mode)))
    else:
        print("Smoothing mode is not selected correctly. Please select one of --unsmoothed --laplace --interpolation")
        exit()

    # write to csv
    sorted_rows = sorted(rows, key = lambda x:x[1])
    outputCSV(sorted_rows, mode, dirPath3)

if __name__ == "__main__":
    main()
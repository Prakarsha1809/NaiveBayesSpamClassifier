# NaiveBayesSpamClassifier
Naive Bayes Spam Classifier in python using the Enron, Ling-spam and Nigerian fraud corpus.

Total 25804 samples with 70% as training and 30% as test samples.

## Functionalities

* Predicts whether entered email is fraud or legitimate
* 98.43% accuracy with 10 fold cross validation
* 96.55% accuracy without cross validation

## Getting Started

* Unzip the ham1 and spam1 files and save them to ham1 and spam1 folders.
* Change the path of all the files used in the code to their current location.(Locations in the code where path should be changed are mentioned in the code)
* The classifier using 10 fold cross validation takes 6 hours or more to train. If you want to use it then uncomment that section and change the following code as:
```
def ex(p):
    egs=preprocess(p)
    example=[egs]
    #example_counts = tfidf.transform(count.transform(example))
    predictions = clf.predict(example)              
```

## Installations
Install the following packages in cmd before use.
```
pip install numpy
pip install matplotlib
pip install pandas
pip install scipy
pip install scikit learn
pip install nltk
pip install python-tk
```
## Steps to use

  * run the .py file

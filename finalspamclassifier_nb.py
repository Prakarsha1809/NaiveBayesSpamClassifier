#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyprind
import pandas as pd
import os
import sys
import numpy as np
import email
from sklearn.model_selection import train_test_split
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from tkinter import*
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


# In[ ]:


# importing data 
pbar=pyprind.ProgBar(16002)
labels={'spam1':1, 'ham1':0}
df=pd.DataFrame()
for l in('ham1','spam1'):
    path='C:/Users/Toshal/Documents/machine_learning/D/%s' %(l)  #path to the folder in which spam1 and ham1 folders are saved
    for file in os.listdir(path):
        with open(os.path.join(path,file),'r',encoding='latin-1') as infile:
            txt=infile.read()
        df=df.append([[txt,labels[l]]], ignore_index=True)
        pbar.update()
df.columns=['Text','Class']
print("Documents accessed successfully.")
df.head()

np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('C:/Users/Toshal/Documents/machine_learning/semifinal1.csv',index=False) #path to save the new .csv file
df=pd.read_csv('C:/Users/Toshal/Documents/machine_learning/semifinal1.csv')  #above same path where you saved the .csv file
print("Successfully saved as .csv file and randomized.")
df.head()


# In[ ]:


#seperating subject from body 
def insert_value(dictionary, key, value):
    if key in dictionary:
        values = dictionary.get(key)
        values.append(value)
        dictionary[key] = values
    else:
        dictionary[key] = [value]
    return dictionary

def get_headers(df, header_names):
    headers = {}
    messages = df["Text"]
    for message in messages:
        e = email.message_from_string(message)
        for item in header_names:
            header = e.get(item)
            insert_value(dictionary = headers, key = item, value = header) 
    print("Successfully retrieved header information!")
    return headers
header_names = ["Subject"]    
headers = get_headers(df, header_names)


def get_messages(df):
    messages = []
    for item in df["Text"]:
        # Return a message object structure from a string
        e = email.message_from_string(item)    
        # get message body  
        message_body = e.get_payload()
        message_body = message_body.lower()
        messages.append(message_body)
    return messages
msg_body = get_messages(df)
df["Text"] = msg_body
print("Successfully retrieved Header and Message body!")


# In[ ]:


df.head()


# In[ ]:


df1=pd.read_csv('C:/Users/Toshal/Documents/machine_learning/fraud_email_.csv') #path where the fraud_email.csv is saved on your computer 


# In[ ]:


df2=pd.read_csv('C:/Users/Toshal/Documents/machine_learning/my_project/combined.csv') #path where the fraud_email.csv is saved on your computer


# In[ ]:


np.random.seed(0)
df2=df2.reindex(np.random.permutation(df2.index))
df2.to_csv('C:/Users/Toshal/Documents/machine_learning/my_project/combined.csv',index=False) #same abpve path of combined.csv
print("Successfully combined and randomized the 2 csv files.")


# In[ ]:


df2.describe()


# In[ ]:


print("The no. of null chanaracters in data:")
df2.isnull().sum()


# In[ ]:


df3=df2.dropna(axis=0)
print("Successfully dropped NaN rows.")
df3.isnull().sum()


# In[ ]:


print("The no. of duplicate rows are:")
df3.duplicated().sum()


# In[ ]:


df3.drop_duplicates(keep='first',inplace=True)
print("Successfully dropped duplicate rows!")
print("The no. of duplicate rows are:")
df3.duplicated().sum()


# In[ ]:


df3.describe()


# In[ ]:


df3.drop(df3.index[df3['Class'] == 'Class'],inplace=True)
print("Sucessfully removed extra Labels.")


# In[ ]:


df3.describe()


# In[ ]:


df3.to_csv('C:/Users/Toshal/Documents/machine_learning/my_project/cleaned.csv',index=False) # path where you want to save the cleaned.csv file
print("Successfully saved the cleaned data in csv file.")


# In[ ]:


df3=pd.read_csv('C:/Users/Toshal/Documents/machine_learning/my_project/cleaned.csv') # same above path of cleaned.csv file


# In[ ]:


y=df3.Class
x=df3.drop('Class',axis=1)
x_train, x_test, y_train, y_test=            train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
print("Successfully split the cleaned data into train and test samples with 7:3 ratio.")
print("Length of training data:",len(x_train))
print("Length of testing data:",len(x_test))


# In[ ]:


sno = nltk.stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
def preprocess(raw_review):
    
    #cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', raw_review)
    
    #cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',cleaned)
    
    cleaned = re.sub(r'£|\$', 'moneysymb', raw_review)
    
    cleaned = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr', cleaned)
    
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    
    cleaned=re.sub('<[^>]','',cleaned)
    
    cleaned=re.sub('[\W]+',' ',cleaned)
    words = [word.lower() for word in cleaned.split()]
    
    # remove punctuation from each word
    
    #table = str.maketrans('', '', string.punctuation)
    #stripped = [w.translate(table) for w in words]
    
    stops = set(stopwords.words("english")) 
    stops.update(('','subject'))

    wordi= [lemmatizer.lemmatize(word) for word in words if word not in stops]
    wordz=[sno.stem(worde) for worde in wordi]
    return( " ".join( wordz )) 
     
print("Successfully preprocessed data.")
print("Example for preprocessing:\n\n</br>? what is your prakarsha-dahat yourself doing here!!? subject no-bro HM IN the light of fire dsjd@jfj.com low lower running and run What's armor-like leaves: THIS IS AN EXAMPLE OF PREPROCESSING!7 473")
dy=preprocess("</br>? what is your prakarsha-dahat yourself doing here!!? subject no-bro HM IN the light of fire dsjd@jfj.com low lower running and run What's armor-like leaves: THIS IS AN EXAMPLE OF PREPROCESSING!7 473") 
print("\n \n Preprocessed data:\n",dy)


# In[ ]:


predata_train=x_train['Text'].apply(preprocess)
print("Successfully preprocessed the training data:")
predata_train.to_csv('C:/Users/Toshal/Documents/machine_learning/my_project/predata_train.csv',index=False)
predata_train.head()


# In[ ]:


predata_test=x_test['Text'].apply(preprocess)
print("Successfully preprocessed the testing data:")

predata_test.head()


# In[ ]:


# method 1 classifier without cross validation
count=CountVectorizer(max_features=8000)
tfidf=TfidfTransformer()
train_fet=tfidf.fit_transform(count.fit_transform(predata_train.values))
train_leb=y_train.values
print("Successfully vectorized training data.")
train_fet


# In[ ]:


classifier = MultinomialNB()
classifier.fit(train_fet, train_leb)
print("Successfully trained the Naive Bayes Classifier.")


# In[ ]:


test_fet=tfidf.transform(count.transform(predata_test.values))
print("Successfully vectorized the testing data.")


# In[ ]:


result1 = classifier.predict(test_fet)
print("Successfully predicted the testing labels on the trained model.")
k=print("Accuracy of our Naive Bayes model:",metrics.accuracy_score(result1, y_test)*100,'%')


# In[ ]:


cm = confusion_matrix(y_test, result1)
P=print("No. of Actual Legitimate & Predicted as Legitimate emails: ",cm[0][0],"\nNo. of Actual Legitimate & Predicted as Fraud emails: ",cm[0][1],"\nNo. of Actual Fraud & Predicted  as Legitimate emails: ",cm[1][0],"\nNo. of Actual Fraud & Predicted as Fraud emails: ",cm[1][1],"\n\n")
P


# In[ ]:


l=print(classification_report(y_test.values, classifier.predict(test_fet), digits=4))
l


# In[ ]:


examples = str(input("Enter the email to classify:"))
egs=preprocess(examples)
example=[egs]
example_counts = tfidf.transform(count.transform(example))
predictions = classifier.predict(example_counts)
                                 
if predictions==1:
    m=print("\n \n The given email is predicted to be Fraud.")
    m
else:
    n=print("\n\nThe given email is predicted to be Legitimate.")
    n


# In[ ]:


#method 2 classfier with cross validtion of 10 folds
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}


# In[ ]:


clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
clf.fit(predata_train.values, y_train.values)


print(classification_report(y_test.values, clf.predict(predata_test.values), digits=4))


# In[ ]:


rel=clf.predict(predata_test.values)
print("Accuracy of our Naive Bayes model:",metrics.accuracy_score(rel, y_test)*100,'%')


# In[ ]:


cm = confusion_matrix(y_test, rel)
print("No. of Actual Legitimate & Predicted as Legitimate emails: ",cm[0][0])
print("No. of Actual Legitimate & Predicted as Fraud emails: ",cm[0][1])
print("No. of Actual Fraud & Predicted  as Legitimate emails: ",cm[1][0])
print("No. of Actual Fraud & Predicted as Fraud emails: ",cm[1][1])


# In[ ]:


examples = str(input("Enter the email to classify:"))
egs=preprocess(examples)
example=[egs]
rels=clf.predict(example)
if rels=='1':
    print("\n\nThe given email is predicted to be Fraud.")
else:
    print("\n\nThe given email is predicted to be Legitimate.")


# In[ ]:


from tkinter import messagebox
root = Tk()
root.title('FRAUD DETECTION SYSTEM')
root.state('zoomed')
f1=Frame(root,height=1000,width=3000,bg='black')
f1.pack()
label=Label(f1,text='FRAUD   DETECTION   SYSTEM',font='Papyrus 35 bold',fg='white',bg='black',pady=1).place(x=320,y=70)
label1=Label(f1,text='Enter Email :',font='Papyrus 25 bold',fg='white',bg='black',pady=1).place(x=380,y=300)

#e1=Entry(f1,height=30,width=25,textvariable=sid).place(x=246,y=40)
textBox=Text(f1,height=14,width=50,font=12)
textBox.place(x=600,y=300)
btn1 = Button(f1, text='Predict Fraud',font='Papyrus 15 bold',fg='black',bg='white',pady=1,command=lambda: retrieve_input()).place(x=700,y=700)
def retrieve_input():
    p=inputValue=textBox.get("1.0","end-1c")
    q=str(p)
    ex(q)
        
                          
def ex(p):
        
        
    egs=preprocess(p)
    example=[egs]
    #example_counts = tfidf.transform(count.transform(example))
    predictions = clf.predict(example)
                                 
    if predictions==1:
        messagebox.showinfo("Prediction","              FRAUD!                 ")
            
    else:
        messagebox.showinfo("Prediction","        LEGITIMATE!           ")
            

root.mainloop()

        
        
    



import os
import re
import itertools
import numpy as np
import pandas as pd

#visual libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from googlesearch import search
from tld import get_tld
from tqdm import tqdm
import time
import os
import threading


df = pd.read_csv('malicious_phish.csv') #using panda library to read the csv file

print(df.shape) #turns our data into rows and columns
df.head() #shows the first 5 entries in the dataset


df.type.value_counts() #this counts the different types of urls in the dataset (it is 4 for our case: benign, defacement, phishing, malware)

#basically splits the data into 4 dataframes (categories) based on their type using the panda library
df_phish   = df[df.type=='phishing']
df_malware = df[df.type=='malware']
df_deface  = df[df.type=='defacement']
df_benign  = df[df.type=='benign']

# # 1) Phishing
# phish_url = " ".join(i for i in df_phish.url)
# wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(phish_url)
# plt.figure(figsize=(12,14), facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show() #this graph shows the most common words in phishing urls

# # # 2) Malware
# malware_url = " ".join(i for i in df_malware.url)
# wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(malware_url)
# plt.figure(figsize=(12,14), facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show() #this graph shows the most common words in malware urls

# # # 3) Defacement
# deface_url = " ".join(i for i in df_deface.url)
# wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(deface_url)
# plt.figure(figsize=(12,14), facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show() #this graph shows the most common words in defacement urls

# # # 4) Benign
# benign_url = " ".join(i for i in df_benign.url)
# wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(benign_url)
# plt.figure(figsize=(12,14), facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show() #this graph shows the most common words in benign urls

#the actual functions that extract features from the URLs:

def having_ip_address(url): #checks for ip in the actual link (usually unsafe urls have some)
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 hex
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # IPv6
    return 1 if match else 0

def abnormal_url(url): #checks the url against its hostname to determine if its abnormal or not
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url) if hostname != 'None' else None
    return 1 if match else 0

def google_index(url): #checks whether the URL is indexed by google (stored in googles database)
    site = search(url, 5)
    return 1 if site else 0

# these functions count specific characters in the URL
def count_dot(url):        
    return url.count('.')
def count_www(url):        
    return url.count('www')
def count_atrate(url):     
    return url.count('@')
def no_of_dir(url):        
    return urlparse(url).path.count('/')
def no_of_embed(url):      
    return urlparse(url).path.count('//')

def shortening_service(url): #uses the re search method to look for shortening services in the URL
    match = re.search(
        'bit\\.ly|goo\\.gl|shorte\\.st|go2l\\.ink|x\\.co|ow\\.ly|t\\.co|tinyurl|tr\\.im|is\\.gd|cli\\.gs|'
        'yfrog\\.com|migre\\.me|ff\\.im|tiny\\.cc|url4\\.eu|twit\\.ac|su\\.pr|twurl\\.nl|snipurl\\.com|'
        'short\\.to|BudURL\\.com|ping\\.fm|post\\.ly|Just\\.as|bkite\\.com|snipr\\.com|fic\\.kr|loopt\\.us|'
        'doiop\\.com|short\\.ie|kl\\.am|wp\\.me|rubyurl\\.com|om\\.ly|to\\.ly|bit\\.do|t\\.co|lnkd\\.in|'
        'db\\.tt|qr\\.ae|adf\\.ly|goo\\.gl|bitly\\.com|cur\\.lv|tinyurl\\.com|ow\\.ly|bit\\.ly|ity\\.im|'
        'q\\.gs|is\\.gd|po\\.st|bc\\.vc|twitthis\\.com|u\\.to|j\\.mp|buzurl\\.com|cutt\\.us|u\\.bb|yourls\\.org|'
        'x\\.co|prettylinkpro\\.com|scrnch\\.me|filoops\\.info|vzturl\\.com|qr\\.net|1url\\.com|tweez\\.me|v\\.gd|'
        'tr\\.im|link\\.zip\\.net',
        url
    )
    return 1 if match else 0

def count_https(url):     #all these methods here count specific characters in the URL
    return url.count('https')
def count_http(url):       
    return url.count('http')
def count_per(url):        
    return url.count('%')
def count_ques(url):       
    return url.count('?')
def count_hyphen(url):     
    return url.count('-')
def count_equal(url):      
    return url.count('=')
def url_length(url):       
    return len(str(url))
def hostname_length(url):  
    return len(urlparse(url).netloc)

def suspicious_words(url): #checks for suspicious words in the URL
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def digit_count(url): #checks the number of digits in the URL
    digits = 0
    for i in url:
        if i.isnumeric():
            digits += 1
    return digits

def letter_count(url): #checks the number of letters in the URL
    letters = 0
    for i in url:
        if i.isalpha():
            letters += 1
    return letters

def fd_length(url): #method checks the length of the first directory in the path (usually malicious URLs have long first directory names)
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld): #method checks the length of the top level domain
    try:
        return len(tld)
    except:
        return -1

#Apply features by creating a dataframe with additional columns with what we call. Lambda is a way to create small functions and apply quicker in pandas than for loops
df['use_of_ip']         = df['url'].apply(lambda i: having_ip_address(i)) 
df['abnormal_url']      = df['url'].apply(lambda i: abnormal_url(i))
df['google_index']      = df['url'].apply(lambda i: google_index(i))
df['count.']            = df['url'].apply(lambda i: count_dot(i))
df['count-www']         = df['url'].apply(lambda i: count_www(i))
df['count@']            = df['url'].apply(lambda i: count_atrate(i))
df['count_dir']         = df['url'].apply(lambda i: no_of_dir(i))
df['count_embed_domian']= df['url'].apply(lambda i: no_of_embed(i))
df['short_url']         = df['url'].apply(lambda i: shortening_service(i))
df['count-https']       = df['url'].apply(lambda i: count_https(i))
df['count-http']        = df['url'].apply(lambda i: count_http(i))
df['count%']            = df['url'].apply(lambda i: count_per(i))
df['count?']            = df['url'].apply(lambda i: count_ques(i))
df['count-']            = df['url'].apply(lambda i: count_hyphen(i))
df['count=']            = df['url'].apply(lambda i: count_equal(i))
df['url_length']        = df['url'].apply(lambda i: url_length(i))
df['hostname_length']   = df['url'].apply(lambda i: hostname_length(i))
df['sus_url']           = df['url'].apply(lambda i: suspicious_words(i))
df['count-digits']      = df['url'].apply(lambda i: digit_count(i))
df['count-letters']     = df['url'].apply(lambda i: letter_count(i))
df['fd_length']         = df['url'].apply(lambda i: fd_length(i))

df['tld']        = df['url'].apply(lambda i: get_tld(i, fail_silently=True))
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))

df = df.drop("tld", axis=1) #drop the tld column as we only need its length

df.columns #shows all the columns in the dataframe including the new features
df['type'].value_counts() #again counts the different types of urls in the dataset


sns.set(style="darkgrid") #setting the style for the graphs

#down below are different graphs showing the distribution of different features across the 4 categories of urls
# 1. use_of_ip 
# ax = sns.countplot(y="type", data=df, hue="use_of_ip")
# plt.show() 
# # # 2. abnormal_url
# ax = sns.countplot(y="type", data=df, hue="abnormal_url")
# plt.show() 

# # # 3. google_index
# ax = sns.countplot(y="type", data=df, hue="google_index")
# plt.show()

# # # 4. short_url
# ax = sns.countplot(y="type", data=df, hue="short_url")
# plt.show()

# # # 5. sus_url
# ax = sns.countplot(y="type", data=df, hue="sus_url")
# plt.show() 

# # # 6. count of dot
# ax = sns.catplot(x="type", y="count.", kind="box", data=df)
# plt.show() 
# # # 7. count-www
# ax = sns.catplot(x="type", y="count-www", kind="box", data=df)
# plt.show() 

# # # 8. count@
# ax = sns.catplot(x="type", y="count@", kind="box", data=df)
# plt.show()

# # # 9. count_dir
# ax = sns.catplot(x="type", y="count_dir", kind="box", data=df)
# plt.show()

# # # 10. hostname length
# ax = sns.catplot(x="type", y="hostname_length", kind="box", data=df)
# plt.show()

# # # 11. first directory length
# ax = sns.catplot(x="type", y="fd_length", kind="box", data=df)
# plt.show()

# # # 12. tld_length
# ax = sns.catplot(x="type", y="tld_length", kind="box", data=df)
# plt.show()

#this makes the data ready for machine learning models by encoding the target labels into numerical values
lb_make = LabelEncoder() #initializes the label encoder
df["type_code"] = lb_make.fit_transform(df["type"]) #encodes the labels
df["type_code"].value_counts() #shows the counts of each encoded label

# (filters out google_index per original because it was not used in the models only to visualize)
X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

y = df['type_code'] 
X.head()
X.columns


# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(    
    X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5
) #80% training and 20% test split function


# 1) Random Forest Classifier
#    URL Features
#         |
#    Random Samples
#     /    |    \
# Tree1  Tree2  Tree3 ... Tree100
#  |      |      |
# Vote   Vote   Vote
#     \   |   /
#      Results
#  (Majority Vote)
print("\nGuide:")
print("Precision: Of all URLs the model marked as malicious, how many were actually bad.")
print("Recall: Of all real malicious URLs, how many the model successfully caught.")
print("F1-score: A balance between precision and recall how consistent and reliable the model is overall.")
print("Support: The number of test samples used for each URL type.")

print("\nTraining Random Forest Model...")
progressbar = tqdm(total=100, desc="In Progress", ncols=80, colour='green')

def train_model():
    global rf, y_pred_rf
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')#initializes the random forest classifier
    rf.fit(X_train, y_train)#fits the model with the training data
    y_pred_rf = rf.predict(X_test)


training_thread = threading.Thread(target=train_model)
training_thread.start()

while training_thread.is_alive():
    progressbar.update(1)
    time.sleep(0.8)
    if progressbar.n >= 100:
        progressbar.n = 0
        progressbar.refresh()

training_thread.join()
progressbar.n = progressbar.total 
progressbar.refresh()
progressbar.close()


print("\nRandom Forest Model Results:")
y_pred_rf = rf.predict(X_test) #makes predictions on the test data

print(classification_report(y_test, y_pred_rf, target_names=['benign', 'defacement', 'phishing', 'malware']))
score = accuracy_score(y_test, y_pred_rf)
print("accuracy:   %0.3f" % score) #prints the accuracy of the model





# cm = confusion_matrix(y_test, y_pred_rf) #creates the confusion matrix
# cm_df = pd.DataFrame(cm,
#                      index = ['benign', 'defacement','phishing','malware'],
#                      columns = ['benign', 'defacement','phishing','malware']) #creates a dataframe for the confusion matrix
# plt.figure(figsize=(8,6))
# sns.heatmap(cm_df, annot=True, fmt=".1f")
# plt.title('Confusion Matrix - RandomForest')
# plt.ylabel('Actual Values')
# plt.xlabel('Predicted Values')
# plt.show() #plots the confusion matrix

# plt.style.use('default')
sns.set_style('whitegrid')
# colors = ["#FF6F61", "#FFB347", "#FFD700", "#90EE90", "#66CDAA", "#1E90FF", "#9370DB", "#FF69B4", "#00CED1", "#FF4500"]

# feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
# feat_importances.sort_values().plot(kind="barh", figsize=(10, 6),
#                                    color=colors * (len(feat_importances) // len(colors) + 1))
# plt.title('Feature Importances - RandomForest', color='black', fontsize=14, weight='bold')
# plt.xlabel('Importance', color='black')
# plt.ylabel('Features', color='black')
# plt.tight_layout()
# plt.show() #plots the feature importance graph


# 2) LightGBM Classifier (leaf growth)
#          Root
#           |
#      Best Split
#        /     \
#    Leaf1    Node2
#              /  \
#         Leaf2  Leaf3
#   
#   Next Tree focuses on errors:
#          Root
#         /    \
#    Better   Better
#    Split    Split
print("\nTraining LightGBM Model...")
progressbar = tqdm(total=100, desc="In Progress", ncols=80, colour='magenta')
def train_lightgbm():
    global LGB_C, y_pred_lgb
    lgb = LGBMClassifier(objective='multiclass', boosting_type='gbdt',
                         n_jobs=5, random_state=5, verbose=-1)
    LGB_C = lgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss"
    )
    y_pred_lgb = LGB_C.predict(X_test)

training_thread = threading.Thread(target=train_lightgbm)
training_thread.start()
while training_thread.is_alive():
    progressbar.update(2)
    time.sleep(0.8)
    if progressbar.n >= 100:
        progressbar.n = 0  # loop around if it reaches 100 early
        progressbar.refresh()

training_thread.join()
progressbar.n = progressbar.total
progressbar.refresh()
progressbar.close()


print("\nLightGBM Model Results:")
print(classification_report(y_test, y_pred_lgb, target_names=['benign', 'defacement','phishing','malware']))
score = metrics.accuracy_score(y_test, y_pred_lgb)
print("accuracy:   %0.3f" % score)

# cm = confusion_matrix(y_test, y_pred_lgb)
# cm_df = pd.DataFrame(cm,
#                      index = ['benign', 'defacement','phishing','malware'],
#                      columns = ['benign', 'defacement','phishing','malware'])
# plt.figure(figsize=(8,6))
# sns.heatmap(cm_df, annot=True, fmt=".1f")
# plt.title('Confusion Matrix - LightGBM')
# plt.ylabel('Actal Values')
# plt.xlabel('Predicted Values')
# plt.show()

# plt.style.use('default')
# sns.set_style('whitegrid')
# colors = ["#FF6F61", "#FFB347", "#FFD700", "#90EE90", "#66CDAA", "#1E90FF", "#9370DB", "#FF69B4", "#00CED1", "#FF4500"]

# feat_importances = pd.Series(lgb.feature_importances_, index=X_train.columns)
# feat_importances.sort_values().plot(kind="barh", figsize=(10, 6),
#                                    color=colors * (len(feat_importances) // len(colors) + 1))
# plt.title('Feature Importances - LightGBM', color='black', fontsize=14, weight='bold')
# plt.xlabel('Importance', color='black')
# plt.ylabel('Features', color='black')
# plt.tight_layout()
# plt.show() #plots the feature importance graph



# 3) XGBoost Classifier
#  First Tree         Residual Tree 1     Residual Tree 2
#     [1]                 [2]                 [3]
#      O                   O                   O        Level 1
#    /   \               /   \               /   \
#   O     O             O     O             O     O    Level 2
#  / \   / \           / \   / \           / \   / \
# O   O O   O         O   O O   O         O   O O   O Level 3
#
# [1]: Initial Predictions
# [2]: Corrects Errors from [1]
# [3]: Corrects Errors from [1]+[2]
# 
# Each new tree focuses on reducing errors from previous predictions
print("\nTraining XGBoost Model...")
progressbar = tqdm(total=100, desc="In Progress", ncols=80, colour='cyan')
def train_xgboost():
    global xgb_c, y_pred_x
    xgb_c = xgb.XGBClassifier(
        n_estimators=100,
        eval_metric="mlogloss",
        verbosity=0,       # turn off built-in print output
        random_state=5
    )
    xgb_c.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_x = xgb_c.predict(X_test)

training_thread = threading.Thread(target=train_xgboost)
training_thread.start()

while training_thread.is_alive():
    progressbar.update(1)
    time.sleep(0.3) 
    if progressbar.n >= 100:
        progressbar.n = 100
        progressbar.refresh()

training_thread.join()
progressbar.n = progressbar.total
progressbar.refresh()
progressbar.close()

print("\nXGBoost Model Results:")
print(classification_report(y_test, y_pred_x, target_names=['benign', 'defacement','phishing','malware']))
score = metrics.accuracy_score(y_test, y_pred_x)
print("accuracy:   %0.3f" % score)

# cm = confusion_matrix(y_test, y_pred_x)
# cm_df = pd.DataFrame(cm,
#                      index = ['benign', 'defacement','phishing','malware'],
#                      columns = ['benign', 'defacement','phishing','malware'])
# plt.figure(figsize=(8,6))
# sns.heatmap(cm_df, annot=True, fmt=".1f")
# plt.title('Confusion Matrix - XGBoost')
# plt.ylabel('Actal Values')
# plt.xlabel('Predicted Values')
# plt.show()

# plt.style.use('default')
# sns.set_style('whitegrid')
# colors = ["#FF6F61", "#FFB347", "#FFD700", "#90EE90", "#66CDAA", "#1E90FF", "#9370DB", "#FF69B4", "#00CED1", "#FF4500"]

# feat_importances = pd.Series(xgb_c.feature_importances_, index=X_train.columns)
# feat_importances.sort_values().plot(kind="barh", figsize=(10, 6),
#                                    color=colors * (len(feat_importances) // len(colors) + 1))
# plt.title('Feature Importances - XGBoost', color='black', fontsize=14, weight='bold')
# plt.xlabel('Importance', color='black')
# plt.ylabel('Features', color='black')
# plt.tight_layout()
# plt.show() #plots the feature importance graph


#function that extracts features from a given URL
def main(url):
    status = []
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))
    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))
    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))
    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url, fail_silently=True)
    status.append(tld_length(tld))
    return status

#function that gets prediction from a given URL using the LightGBM model
def get_prediction_from_url(test_url):
    features_test = main(test_url)
    features_test = np.array(features_test).reshape((1, -1))
    features_test = pd.DataFrame(features_test, columns=X_train.columns)
    pred = LGB_C.predict(features_test)
    if int(pred[0]) == 0:
        return "SAFE"
    elif int(pred[0]) == 1.0:
        return "DEFACEMENT"
    elif int(pred[0]) == 2.0:
        return "PHISHING"
    elif int(pred[0]) == 3.0:
        return "MALWARE"

#Same demo predictions as humans
urls = ["http://grasslandhotel.com.vn/index.php/component/djcatalog2/items/6-services",
         "https://docs.google.com/spreadsheet/viewform?formkey=dGg2Z1lCUHlSdjllTVNRUW50TFIzSkE6MQ",
         "ftvdb.bfi.org.uk/sift/title/171844",
         "http://www.pashminaonline.com/pure-pashminas",
         "http://www.raci.it/component/user/reset.html",
         "http://asociacionkaribu.org/index.php/servicios/centro-jambo",
         "insidelocation.ga",
         "http://zoetekroon.nl/wp-content/themes/simplo/js/jquery-1.4.2.min.js",
         "newyors.com",
         "http://www.vilagnomad.com/tables/signature-loans-no-credit-check.php",
        ]
print("\n__________ URL Detector Results__________\n")
for url in urls:
    print(f"{url} → {get_prediction_from_url(url)}")

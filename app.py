#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install google-auth google-auth-oauthlib google-auth-httplib2')
get_ipython().system('pip install demoji')
get_ipython().system('pip install pandas')
get_ipython().system('pip install langdetect')


# In[4]:


from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import demoji
from langdetect import detect
import re   # regular expression
from textblob import TextBlob
from sklearn import metrics
# from mlxtend.plotting import plot_confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import pickle 
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
# from google.colab import drive


# In[5]:


CLIENT_SECRETS_FILE = "/content/drive/My Drive/Colab Notebooks/Youtube API/client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


# In[10]:


def get_authenticated_service():
  flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
  credentials = flow.run_console()
  return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)


# In[12]:


drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Colab Notebooks/Youtube API')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
service = get_authenticated_service()


# In[ ]:

query = st.text_input("Please copy and past the name of video on YouTube here: ", Grab the tomatoes before they drop, and make something delicious - ruby ketchup)
query = "How the Grinch Stole Christmas (2/9) Movie CLIP - Baby Grinch (2000) HD"


# PIPLINE ==========================

# In[93]:


query_results = service.search().list(part = 'snippet',q = query,
                                      order = 'relevance', 
                                      type = 'video',
                                      relevanceLanguage = 'en',
                                      safeSearch = 'moderate').execute()

video_id = []
channel = []
video_title = []
video_desc = []
for item in query_results['items']:
    video_id.append(item['id']['videoId'])
    channel.append(item['snippet']['channelTitle'])
    video_title.append(item['snippet']['title'])
    video_desc.append(item['snippet']['description'])


video_id = video_id[0]
channel = channel[0]
video_title = video_title[0]
video_desc = video_desc[0]



video_id_pop = []
channel_pop = []
video_title_pop = []
video_desc_pop = []
comments_pop = []
comment_id_pop = []
reply_count_pop = []
like_count_pop = []


comments_temp = []
comment_id_temp = []
reply_count_temp = []
like_count_temp = []


nextPage_token = None

while 1:
  response = service.commentThreads().list(
                    part = 'snippet',
                    videoId = video_id,
                    maxResults = 100, 
                    order = 'relevance', 
                    textFormat = 'plainText',
                    pageToken = nextPage_token
                    ).execute()


  nextPage_token = response.get('nextPageToken')
  for item in response['items']:
      comments_temp.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
      comment_id_temp.append(item['snippet']['topLevelComment']['id'])
      reply_count_temp.append(item['snippet']['totalReplyCount'])
      like_count_temp.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
      comments_pop.extend(comments_temp)
      comment_id_pop.extend(comment_id_temp)
      reply_count_pop.extend(reply_count_temp)
      like_count_pop.extend(like_count_temp)
        
      video_id_pop.extend([video_id]*len(comments_temp))
      channel_pop.extend([channel]*len(comments_temp))
      video_title_pop.extend([video_title]*len(comments_temp))
      video_desc_pop.extend([video_desc]*len(comments_temp))

  if nextPage_token is  None:
    break



output_dict = {
        'Channel': channel_pop,
        'Video Title': video_title_pop,
        'Video Description': video_desc_pop,
        'Video ID': video_id_pop,
        'Comment': comments_pop,
        'Comment ID': comment_id_pop,
        'Replies': reply_count_pop,
        'Likes': like_count_pop,
        }

output_df = pd.DataFrame(output_dict, columns = output_dict.keys())


duplicates = output_df[output_df.duplicated("Comment ID")]


unique_df = output_df.drop_duplicates(subset=['Comment'])

comments = unique_df

demoji.download_codes()

comments['clean_comments'] = comments['Comment'].apply(lambda x: demoji.replace(x,""))

comments['language'] = 0

count = 0
for i in range(0,len(comments)):


  temp = comments['clean_comments'].iloc[i]
  count += 1
  try:
    comments['language'].iloc[i] = detect(temp)
  except:
    comments['language'].iloc[i] = "error"


comments[comments['language']=='en']['language'].value_counts()

english_comm = comments[comments['language'] == 'en']

en_comments = english_comm

regex = r"[^0-9A-Za-z'\t]"

copy = en_comments.copy()


copy['reg'] = copy['clean_comments'].apply(lambda x:re.findall(regex,x))
copy['regular_comments'] = copy['clean_comments'].apply(lambda x:re.sub(regex,"  ",x))


dataset = copy[['Video Title','Video ID','Comment ID','Replies','Likes','regular_comments']].copy()


dataset = dataset.rename(columns = {"regular_comments":"comments"})


# SENTIMENTAL ANALYSIS

# In[90]:


data = dataset


data['polarity'] = data['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)

data = data.sample(frac=1).reset_index(drop=True)

data['pol_cat']  = 0

data['pol_cat'][data.polarity > 0] = 1
data['pol_cat'][data.polarity <= 0] = -1

data_pos = data[data['pol_cat'] == 1]
data_pos = data_pos.reset_index(drop = True)

data_neg = data[data['pol_cat'] == -1]
data_neg = data_neg.reset_index(drop = True)


data['comments'] = data['comments'].str.lower()


data['comments'][0].strip()


nltk.download("stopwords")
nltk.download("punkt")


stop_words = set(stopwords.words('english'))

data['comments'] = data['comments'].str.strip()

train = data.copy()


def remove_stopwords(line):
    word_tokens = word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


vect_loaded = pickle.load(open('/content/drive/My Drive/Colab Notebooks/Youtube API/vect.pkl', 'rb'))


data['stop_comments'] = data['comments'].apply(lambda x : remove_stopwords(x))
X_test = data['stop_comments']


tf_test = vect_loaded.transform(X_test)

model_loaded = pickle.load(open('/content/drive/My Drive/Colab Notebooks/Youtube API/lr.pkl', 'rb'))

predicted = model_loaded.predict(tf_test)


# In[94]:


data_pos =0
data_neg = 0

for i in range(0,len(predicted)):
  if (predicted[i] == 1):
     data_pos = data_pos + 1
  else:
    data_neg = data_neg + 1


if (data_pos)>= (len(predicted)/3):
  print ("Positive comments")
elif (data_pos)== (len(predicted)/2):
  print ("Not positive and Not negetive comments")
else:
  print("Negetive comments")


# In[16]:





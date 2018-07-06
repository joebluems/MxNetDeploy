import requests
import json
import cv2
import sys
import tweepy

#### twitter setup ####
def get_api(cfg):
  auth = tweepy.OAuthHandler(cfg['consumer_key'], cfg['consumer_secret'])
  auth.set_access_token(cfg['access_token'], cfg['access_token_secret'])
  return tweepy.API(auth)

cfg = { 
    "consumer_key"        : "XXX",
    "consumer_secret"     : "XXX",
    "access_token"        : "XXX",
    "access_token_secret" : "XXX" 
    }
api = get_api(cfg)

### hard coding URL ###
addr = 'http://jblue1:5003'
test_url = addr + '/api/test'

### prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

### read image and send request ###
img = cv2.imread(sys.argv[1])
_, img_encoded = cv2.imencode('.jpg', img)
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

### handle request
print(response.json())

### send tweet results ###
if sys.argv[2]=='1':
  tweet = response.json()
  image = sys.argv[1]
  status = api.update_with_media(image, status=tweet) 


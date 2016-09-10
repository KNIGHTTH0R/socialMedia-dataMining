from __future__ import division
import sys, csv
try:
    import json
except ImportError:
    import simplejson as json
#using the data we fetched in US 
tweets_filename = 'us10KTweets.txt'
tweets_file = open(tweets_filename, 'r')
#sum is the variable to indicate the number of geotagged tweets in US
#difLangUS is a dictionary, the keys are the langid, the values are the number of the tweets with the same langid
#singLangPer is a dictionary, the keys are the langid, the values are the percentage of  this langid tweets
sum = 0
difLangUS = dict()
singLangPer = dict()
#the geotagged tweets and the number  of each langid tagged in geotagged tweets
for line in tweets_file:
    try:
        tweets = json.loads(line.strip())
        if 'text' in tweets and tweets['place'] and tweets['place']['bounding_box']:
            if tweets['place'].get('id', 0) and tweets['place']['bounding_box'].get('coordinates', 0):
               sum += 1
        if tweets['lang']:
            key = tweets['lang']
            difLangUS[key] = difLangUS.get(key, 0) + 1
    except:
        continue
#percentage of each langid tagged in geotagged tweets
for keys in difLangUS.keys():                                                                                             
    singLangPer[keys] = difLangUS.get(keys,0)/8.82 
print "Percentage of each language is in the following:"
for (k,v) in  singLangPer.items():                                                                                            
    print "singLangPer[%s]=" % k,v
#saving the results for problem 5 to draw the picture
newTuple = sorted(singLangPer.items(), key=lambda d:d[1], reverse = True)
with open('usGeo.txt','w') as out:
     csv_out=csv.writer(out)
     for row in newTuple:
         csv_out.writerow(row)
#percentage of geotagged tweets, 882 is the total number of tweets that fetched in US
perTagTweets = sum / 8.82
print "Percentage of Geotagged tweets are %f %%" %  perTagTweets

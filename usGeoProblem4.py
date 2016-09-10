from __future__ import division
import sys, csv
try:
    import json
except ImportError:
    import simplejson as json

tweets_filename = 'us10KTweets.txt'
tweets_file = open(tweets_filename, 'r')
#import pdb
#pdb.set_trace()
sum = 0
difLangUS = dict()
singLangPer = dict()
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
langCountSum = 0
for i in difLangUS.values(): 
    langCountSum +=i
        
for keys in difLangUS.keys():                                                                                             
    singLangPer[keys] = difLangUS.get(keys,0)/langCountSum 
print "Percentage of each language is in the following:"
for (k,v) in  singLangPer.items():                                                                                            
    print "singLangPer[%s]=" % k,v
newTuple = sorted(singLangPer.items(), key=lambda d:d[1], reverse = True)
with open('file.csv','w') as out:
     csv_out=csv.writer(out)
     for row in newTuple:
         csv_out.writerow(row)
perTagTweets = sum / 8.82
print "Percentage of Geotagged tweets are %f %%" %  perTagTweets

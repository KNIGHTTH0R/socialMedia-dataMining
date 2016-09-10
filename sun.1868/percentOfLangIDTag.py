from __future__ import division
import sys, csv
import langid
try:
    import json
except ImportError:
    import simplejson as json
#reading the tweets data file
tweets_filename = 'twitter_streaming_10K_tweets.txt'
tweets_file = open(tweets_filename, "r")
#we need the dictionaries to save the data 
#langCountSum is to sum the LangID tagged tweets
#singLangCount is a dictionary, the keys are the language id , the values are the number of tweets which have the same language id
#singLangPer is a dictionary, keys are the language id, the values are the percentage of language id tagged tweets
#langDic is a dictionary, keys are the language id checked by langid, the values are the numbers of tweets which have the same language
langCountSum = 0
singLangCount = dict()  
singLangPer = dict()
langDic = dict()

for line in tweets_file:
        try:
            tweet = json.loads(line.strip())
            if 'text' in tweet:
                #finding the langid tagged tweets and save each langid and the number of corresponding tweets 
                if tweet['lang']: 
                    langCountSum += 1
                key = tweet['lang']
                singLangCount[key] = singLangCount.get(key,0) + 1
                #finding the tweets with different language checked by langid.py  
                textContent = tweet['text']
                langTuple = langid.classify(textContent)                                                                          
                langkey = langTuple[0]
                langDic[langkey] = langDic.get(langkey,0) + 1
        except:
            continue
#percentage of langid tagged tweets by twitter API        
percentLang = langCountSum/100
#different language tags provided by twitter
difLangCount = len(singLangCount)
#percentage of tweets with different language id checked by twitter API 
for keys in singLangCount.keys():
    singLangPer[keys] = singLangCount.get(keys,0)/langCountSum * 100
#percentage of tweets with different language checked by langid.py
for keyss in langDic.keys():
    langDic[keyss] = langDic.get(keyss, 0)/100
print "Percentage of LangID tagged tweets is %f %% "  %  percentLang
print "Number of different Language %d"  % difLangCount
print "Different language and their corresponding percentage by twitter API is: " 
for (k,v) in  singLangPer.items(): 
    print "singLangPer[%s]=" % k,v 
print "Different language and their corresponding percentage checked by langid is:"
for (k,v) in langDic.items():
    print "langDic[%s]=" % k,v

#comDict dictionary can save the language id and the corresponding percentage, which are got by twitter API and langid.py above
combDict = dict()
for key in singLangPer.keys():
    combDict[key] = [singLangPer[key]]
    if key not in langDic.keys():
        combDict[key].append(0.0)
for key in langDic.keys():
    if key not in combDict.keys():
        combDict[key] = [0.0]
    combDict[key].append(langDic[key]) 
print "Percentage of different language checked by API and langid is in the following:"    
for (k,v) in combDict.items():
    print "combDict[%s]=" % k,v
#saving the results got by twitter API and langid.py for the problem 5 to draw the picture
newTuple = combDict.items()
with open('APIandLangID.txt', 'w') as out:
    csv_out = csv.writer(out)
    for row in newTuple:
        csv_out.writerow(row)


langIdSum = len(langDic)
print "Number of different tagged language checked by langid.py is %d" % langIdSum
print "Languages that twitter API disagree are in the following:"
for langKeyCommon in langDic.keys():
    if singLangCount.has_key(langKeyCommon):
       continue;
    else:
        print "%s" % langKeyCommon

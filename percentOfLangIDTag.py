from __future__ import division
import sys, csv
import langid
try:
    import json
except ImportError:
    import simplejson as json

tweets_filename = 'twitter_streaming_10K_tweets.txt'
tweets_file = open(tweets_filename, "r")

langCountSum = 0
singLangCount = dict()  
singLangPer = dict()
langDic = dict()

for line in tweets_file:
        try:
            tweet = json.loads(line.strip())
            if 'text' in tweet:
                    langCountSum += 1
                    key = tweet['lang']
                    singLangCount[key] = singLangCount.get(key,0) + 1
                    # langid check
                    textContent = tweet['text']
                    langTuple = langid.classify(textContent)                                                                          
                    langkey = langTuple[0]
                    #langDic.setdefault(langkey,)
                    langDic[langkey] = langDic.get(langkey,0) + 1
        except:
            continue
        
percentLang = langCountSum/100
difLangCount = len(singLangCount)
langCountSum = 0
langDicSum = 0
for i in singLangCount.values(): 
    langCountSum +=i
for j in langDic.values():
    langDicSum +=j
for keys in singLangCount.keys():
    singLangPer[keys] = singLangCount.get(keys,0)/langCountSum * 100
for keyss in langDic.keys():
    langDic[keyss] = langDic.get(keyss, 0)/langDicSum *100
print "Percentage of LangID tagged tweets is %f %% "  %  percentLang



print "Number of different Language %d"  % difLangCount
print "Different language and their corresponding percentage by twitter API is: " 
for (k,v) in  singLangPer.items(): 
    print "singLangPer[%s]=" % k,v 
print "Different language and their corresponding percentage checked by langid is:"
for (k,v) in langDic.items():
    print "langDic[%s]=" % k,v


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

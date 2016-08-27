from twitter import *

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))

user = "cocoweixu"
results = twitter.statuses.user_timeline(screen_name = user)

for status in results:
    print "@%s %s" % (user, status["text"])
    retweets = twitter.statuses.retweets._id(_id = status["id"])
    for retweet in retweets:
        print "- retweeted by %s " % (retweet["user"]["screen_name"])

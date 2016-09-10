from twitter import *

config = {}
execfile("config.py", config)

twitter = Twitter(auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))

statuses = twitter.statuses.home_timeline(count = 50) 
for status in statuses:
    print "(%s), @%s %s" % (status["created_at"], status["user"]["screen_name"], status["text"])



from twitter import *

config = {}
execfile("config.py", config)

auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"])
stream = TwitterStream(auth = auth, secure = True)

tweet_iter = stream.statuses.filter(track = "social")

for tweet in tweet_iter:
	#-----------------------------------------------------------------------
	# print out the contents, and any URLs found inside
	#-----------------------------------------------------------------------
	print "(%s) @%s %s" % (tweet["created_at"], tweet["user"]["screen_name"], tweet["text"])
	for url in tweet["entities"]["urls"]:
		print " - found URL: %s" % url["expanded_url"]

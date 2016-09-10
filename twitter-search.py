
from twitter import *

config = {}
execfile("config.py", config)

twitter = Twitter(
		        auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))

query = twitter.search.tweets(q = "lazy dog")

print "Search complete (%.3f seconds)" % (query["search_metadata"]["completed_in"])

for result in query["statuses"]:
	print "(%s) @%s %s" % (result["created_at"], result["user"]["screen_name"], result["text"])

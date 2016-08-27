from twitter import *

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))

source = "cocoweixu"
target = "alan_ritter"
result = twitter.friendships.show(source_screen_name = source, target_screen_name = target)
following = result["relationship"]["target"]["following"]
follows = result["relationship"]["source"]["followed_by"]
print "%s following %s: %s" % (source, target, follows)
print "%s following %s: %s" % (target, source, following)


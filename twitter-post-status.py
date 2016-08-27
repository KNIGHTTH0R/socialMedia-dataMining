from twitter import *

config = {}
execfile("config.py", config)

twitter = Twitter(auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))
new_status = "testing testing"
results = twitter.statuses.update(status = new_status)
print "updated status: %s" %  new_status

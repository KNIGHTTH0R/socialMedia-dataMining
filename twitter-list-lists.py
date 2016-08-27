from twitter import *

config = {}
execfile("config.py", config)
twitter = Twitter(auth=OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))

users = ["alan_ritter", "cocoweixu", "weisun91"]
import pprint
for user in users:
    print "@%s" % (user)
    result = twitter.lists.list(screen_name = user)
    for list in result:
        print " - %s (%d members)" % (list["name"], list["member_count"])



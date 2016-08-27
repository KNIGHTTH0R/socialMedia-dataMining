from twitter import  *

config = {}
execfile("config.py" , config)
twitter = Twitter(auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))


username = "cocoweixu"

query = twitter.friends.ids(screen_name = username)

print "found %d friends" % (len(query["ids"]))

for n in range(0, len(query["ids"]), 100):
	ids = query["ids"][n:n+100]
        subquery = twitter.users.lookup(user_id = ids)
	for user in subquery:
		print " [%s] %s" % ("*" if user["verified"] else " ", user["screen_name"])

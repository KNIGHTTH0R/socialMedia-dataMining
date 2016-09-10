from twitter import *
import sys, csv
config = {}
execfile("config.py", config)

twitter = Twitter(auth = OAuth(config["ACCESS_TOKEN"], config["ACCESS_SECRET"], config["CONSUMER_KEY"], config["CONSUMER_SECRET"]))

latitude =  89.234
longitude = -0.03232
max_range = 1 
num_results = 500
outfile = "output.csv"

csvfile = file(outfile, "w")
csvwriter = csv.writer(csvfile)
row = ["user", "text", "latitude", "longitude"]
csvwriter.writerow(row)
result_count = 0
last_id = None
while result_count < num_results:
 #   import pdb
  #  pdb.set_trace()
    query = twitter.search.tweets(q="",  geocode = "%f, %f, %dkm" % (latitude, longitude, max_range),count =100 ,max_id=last_id) 

    
    for result in query["statuses"]:
        if result ["geo"]:
            user = result["user"]["screen_name"]
            text = result["text"]
            text = text.encode('ascii', 'replace')
            latitude = result["geo"]["coordinates"][0]
            longitude = result["geo"]["coordinates"][1]
            row = [user, text, latitude, longitude]
            csvwriter.writerow(row)
            result_count +=1
        last_id = result["id"]
    print "got %d results" % result_count
csvfile.close()
print "written to %s" % outfile

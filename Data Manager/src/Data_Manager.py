# Goal: Simply read data from redis and output to the console.
# Eventually will log to a database.

############# Imports ##################################
import redis
import json
import time
from datetime import date, datetime
import os
print('Starting up Data Manager...')
############# Redis Setup ##############################
# Connect to the DataStore
REDIS = redis.Redis(host='data_store')

# To setup the queue
queue = REDIS.pubsub()
# Subscribe to the event queue
queue.subscribe('event_queue')
print('Redis Connected...')
############## Execute ##################################
while True:
    # Get message
    next_message = queue.get_message()
    #next_message = json.loads(queue.get_message()['data'].decode())
    if next_message:
        print('------ REDIS Message -------')
        event_time = datetime.now()
        print(event_time)
        print(next_message['data'])
        # Ignore the initial 1 or 2 that comes out of the queue.

    time.sleep(1)

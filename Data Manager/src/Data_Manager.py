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

session = '0'
while True:
    # Get message
    next_message = queue.get_message()

    if next_message:

        try:
            payload = next_message['data'].decode()
            payload_data = json.loads(payload)
            # check which queue
            channel = next_message['channel'].decode()

            if channel == 'event_queue':

                if payload_data['session'] != session:
                    print("-------------------------- Session ---------------------------")
                    session = payload_data['session']
                    print(session)
                event_time = datetime.now()
                print(event_time)
                print(payload_data['message'])
        except:
            pass

    time.sleep(1)

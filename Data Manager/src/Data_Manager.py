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

    session = '0'
    if next_message:
        #print('-------------- REDIS MEssage')
        # print(next_message)
        try:
            payload = next_message['data'].decode()
            payload_data = json.loads(payload)
            # check which queue
            channel = next_message['channel'].decode()
            # print(channel)

            if channel == 'event_queue':
                # print(payload)
                # print(payload_data['session'])
                #payload_session = paload_data['session']
                # print(payload_session)
                if payload_data['session'] != session:
                    print("------------------ Session -------------------")
                    print(session)
                    global session
                    session = payload_data['session']
                event_time = datetime.now()
                print(event_time)
                print(payload_data['message'])
        except:
            pass

    time.sleep(1)

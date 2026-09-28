ACTIVITY_MAP = {
    "0": "bedroom personal activity",
    "1": "using bathroom",
    "2": "preparing breakfast",             
    "3": "grabing food or drinks",
    "4": "resting on the couch or the chairs",
    "5": "having breakfast",     
    "6": "washing dishes",
    "7": "watching TV",                      
    "8": "reading books",
    "9": "drinking water, coffee or drinks",                
    "10": "cleaning appartment",
    "11": "taking shower",                
    "12": "playing video games",
    "13": "getting out of the apartment",          
    "14": "coming into the apartment",
    "15": "chatting/using the cell-phone",            
    "16": "having fruits or snacks",
    "17": "using pc",      
    "18": "preparing dinner",
    "19": "having dinner",     
    "20": "playing board games or puzzles",
    "21": "others", 
}

SENSOR_TYPES = ["appliance", "door_sensor", "location"]

PROMPT_TEXT = """ You are a professional human activity detetor. An eledrly person is living in a house incorporated with PIR sensors for detecting location, 
    smart plugs to detect the activated appliances, and door sensors. 
    You will recieve a sequence of 10 sensor events (On/OFF) and the detected activity of the person associated with the sensors. 
    Your task is to detect the next activity of the person based on the sequence of events from one of the activity labels in the following list.

    0. bedroom personal activity,                     
    1. using bathroom,
    2. preparing breakfast,             
    3. grabing food or drinks,
    4. resting on the couch or the chairs,
    5. having breakfast,                     
    6. washing dishes,
    7. watching TV,                      
    8. reading books,
    9. drinking water, coffee or drinks,                
    10. cleaning appartment,
    11. taking shower,                
    12. playing video games,
    13. getting out of the apartment,          
    14. coming into the apartment,
    15. chatting/using the cell-phone,            
    16. having fruits or snacks,
    17. using pc,                   
    18. preparing dinner,
    19. having dinner,     
    20. playing board games or puzzles,
    21. others, 

    The sensor event sequence and the relevamt activities are: 
    {event_sequence}

    The next sensor event is:
    {event}

    Please make sure to output only the number associated with the predefined activities without any word or symbol.
    """
"""
This module contains the MuRAL (H002) household profile document.
"""
from langchain_core.documents import Document

HOUSEHOLD_ID = "H002"

household_documents = [

    Document(
        page_content="""
The apartment is a 60-square-meter smart-home environment.
It contains an entrance, kitchen, dining room, living room,bathroom, Bedroom 1, and Bedroom 2.
The main entrance is located beside the kitchen and is the only entry and exit point of the apartment.
The kitchen contains a countertop, sink, cabinets, refrigerator, coffee machine, induction stove, range hood, microwave, and kettle.
The dining room contains a dining table with chairs and a food cabinet.
The living room contains a couch, coffee table, television, game console, and bookshelf.
The bathroom contains a sink, toilet, and shower.
The apartment contains two bedrooms: one double bedroom and one single bedroom. 
The bedrooms do not contain internal activity sensors; only their doors are monitored.
""",
        metadata={
            "knowledge_type": "household",
            "household_id": HOUSEHOLD_ID,
            "category": "house_layout"
        }
    ),


    Document(
        page_content="""
The apartment is instrumented with ambient sensors located in specific rooms and activity zones.
Entrance:
- main door sensor: magnetic contact sensor on the apartment's main entrance door. 
Kitchen:
- hood wattmeter
- stove wattmeter
- coffee machine wattmeter
- microwave oven wattmeter
- kettle wattmeter
- counter motion sensor
- cabinet door sensor
- fridge door sensor
- sink motion sensor

The kitchen counter motion sensor monitors activity around the countertop.
The kitchen sink motion sensor monitors activity around the sink.
The cabinet and refrigerator sensors detect opening and closing of their respective doors.

Dining room:
- chair_1 contact sensor
- chair_2 contact sensor
- chair_3 contact sensor
- food cabinet door sensor

The three chair contact sensors monitor use of chairs positioned around the dining table. 
The food cabinet sensor detects opening and closing of the food cabinet located next to the dining table.

Living room:
- bookshelf motion sensor
- TV wattmeter
- game console wattmeter
- couch motion sensor

The bookshelf motion sensor monitors interactions with books, board games, and other objects on the shelves. 
The couch motion sensor is located on the coffee table and is oriented toward the couch to detect activity in the couch area.

Bathroom:
- toilet contact sensor
- global motion sensor
- sink motion sensor

The bathroom global motion sensor monitors movement throughout the bathroom. 
The sink motion sensor specifically monitors activity around the bathroom sink. 
The toilet contact sensor monitors use of the toilet.

Bedrooms:
- bedroom_1 door sensor
- bedroom_2 door sensor

Both bedroom sensors are magnetic contact sensors that detect opening and closing of the bedroom doors. 
There are no internal activity sensors inside either bedroom.
""",
        metadata={
            "knowledge_type": "household",
            "household_id": HOUSEHOLD_ID,
            "category": "sensor_configuration"
        }
    ),
]

"""
This module contains general knowledge documents related to human activity recognition (HAR) principles, sensor reasoning, appliance reasoning, and specific activity rules. 
These documents provide contextual information and guidelines for interpreting sensor data in the context of HAR.
"""

from langchain_core.documents import Document

general_documents = [
    Document(
        page_content="""
Human activity recognition should be based on multiple sources of
evidence rather than a single sensor event.

PIR sensors provide evidence about occupancy, movement, and probable
location, but PIR activation alone does not uniquely identify an
activity.

Active appliances provide contextual evidence about activities.
For example, activation of cooking appliances while the resident is
present in the kitchen increases the likelihood of food preparation.

Temporal information is important. Activity recognition should consider
sequences of sensor events within a time window rather than interpreting
every event independently.

When evidence is ambiguous, the system should maintain uncertainty
rather than infer an activity from weak evidence.
""",
        metadata={
            "knowledge_type": "general",
            "category": "har_principle",
            "activity": "all"
        }
    ),

# sensor_reasoning
    Document(
        page_content="""
A PIR sensor detects motion within its sensing area.

Repeated PIR activation in the same room suggests sustained presence or
repeated movement in that location.

Sequential activation of PIR sensors in neighboring rooms can indicate
movement between those rooms.

A short isolated PIR activation may represent movement through a room
rather than an activity being performed there.

PIR inactivity does not necessarily mean that the resident is absent.
Activities such as sleeping, watching television, reading, or sitting
may produce little movement.

If pets are present, PIR activation alone should not automatically be
interpreted as human movement.
""",
        metadata={
            "knowledge_type": "general",
            "category": "sensor_reasoning",
            "sensor": "PIR",
            "activity": "all"
        }
    ),

# appliance reasoning
    Document(
        page_content="""
Appliance activation provides contextual evidence about human activity,
but an active appliance does not necessarily mean that the resident is
currently interacting with it.

Cooking appliances such as an oven, microwave, toaster, or kettle
increase the probability of food preparation when combined with
kitchen occupancy.

Television activity combined with sustained presence in the living room
increases the probability of watching television.

Washing-machine activation may indicate laundry-related activity,
especially around the beginning and end of the appliance cycle.

Appliances that operate automatically or for long periods should not be
treated as continuous evidence of human interaction.

NILM appliance detections are probabilistic. Low-confidence appliance
detections should contribute less strongly to activity inference than
high-confidence detections.
""",
        metadata={
            "knowledge_type": "general",
            "category": "sensor_reasoning",
            "sensor": "appliance",
            "activity": "all"
        }
    ),


# cooking
    Document(
        page_content="""
Cooking or meal preparation is more likely when the resident is located
in the kitchen and one or more food-preparation appliances are active.

Relevant appliances may include an oven, microwave, kettle, toaster,
stove, or coffee maker.

Repeated or sustained kitchen PIR activity strengthens the evidence for
cooking.

A sequence involving entering the kitchen, activating cooking
appliances, remaining in the kitchen, and later moving toward a dining
area is consistent with meal preparation.

Kitchen presence without cooking-appliance activity may represent other
activities such as obtaining food, cleaning, or passing through the
kitchen.

Time of day may provide supporting evidence but should not override
stronger sensor evidence.
""",
        metadata={
            "knowledge_type": "general",
            "category": "activity_rule",
            "activity": "cooking",
        }
    ),


# watching TV
    Document(
        page_content="""
Watching television is more likely when the television is active and
the resident is located in the room containing the television.

Sustained presence with relatively little PIR activity may still be
consistent with watching television because the resident may remain
seated.

Television activation alone is insufficient evidence because the
television may remain active when the resident leaves the room.

Television state, resident location, duration, and recent movement
should be considered together.
""",
        metadata={
            "knowledge_type": "general",
            "category": "activity_rule",
            "activity": "watching_tv",
        }
    ),


# doing the laundary
    Document(
        page_content="""
Laundry activity may be associated with interaction with a washing
machine or dryer.

The beginning of washing-machine operation combined with resident
presence near the appliance provides evidence of starting laundry.

Continuous washing-machine operation should not be interpreted as
continuous laundry activity because the resident may leave while the
machine operates.

Resident presence near the washing machine near the end of its cycle
may indicate unloading or transferring laundry.

Laundry recognition should emphasize interaction periods rather than
the entire appliance operating period.
""",
        metadata={
            "knowledge_type": "general",
            "category": "activity_rule",
            "activity": "laundry"
        }
    ),
]

# using bathroom
Document(
    page_content="""
Using the bathroom is more likely when the resident enters the bathroom
and remains there for a period of time.

Bathroom PIR activation provides evidence that the resident may be
using the bathroom, especially when the preceding sensor sequence
indicates movement from another room through the hallway toward the
bathroom.

Sustained or repeated bathroom PIR activity strengthens the evidence
for bathroom use. However, a short isolated bathroom PIR activation
may represent briefly entering the bathroom or passing near the
sensor rather than performing a bathroom-related activity.

Bathroom use generally does not require detectable electrical appliance
activity. Therefore, the absence of NILM-detected appliance activity
should not reduce the likelihood of bathroom use when location and
motion evidence strongly support it.

If a pet is present in the household, isolated bathroom PIR activation
should be interpreted cautiously. Recent human location, movement
between adjacent rooms, duration of bathroom occupancy, and other
sensor observations should be considered when determining whether the
PIR activation was caused by the resident.

The resident's typical bathroom-use patterns and time of day may provide
additional contextual evidence, but they should not override current
sensor observations.

Evidence for bathroom use is strongest when the temporal sensor sequence
shows movement toward the bathroom, bathroom PIR activation, bathroom door opening, or sustained
bathroom presence, followed by movement away from the bathroom.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "using_bathroom",
    }
)

# bedroom personal activity'
Document(
    page_content="""
Bedroom personal activity refers to personal activities performed by the
resident in the bedroom that cannot be reliably classified into a more
specific activity from the available sensor observations.

Bedroom PIR activation or sustained presence in the bedroom provides
the primary location evidence for bedroom personal activity.

Repeated bedroom PIR activation may indicate that the resident is
moving around the bedroom while performing a personal activity.
However, limited PIR activity does not rule out bedroom personal
activity because some personal activities involve little movement.

A transition from another area of the house into the bedroom followed
by sustained bedroom presence strengthens the evidence for bedroom
personal activity.

Bedroom personal activity generally does not require a specific
electrical appliance. Therefore, the absence of NILM-detected appliance
activity should not reduce the likelihood of this activity when the
location and temporal evidence support it.

If bedroom-associated appliances or devices are active, they may provide
supporting context but should not automatically determine the activity.

Bedroom personal activity should be distinguished from sleeping.
Extended bedroom presence with very little movement, particularly during
the resident's typical sleep period, may provide stronger evidence for
sleeping. More frequent movement or shorter periods of bedroom occupancy
may provide stronger evidence for bedroom personal activity.

Bedroom personal activity should also be distinguished from simply
passing through or briefly entering the bedroom. A very short isolated
bedroom PIR activation provides weaker evidence than sustained or
repeated bedroom presence.

If a pet is present in the household, isolated bedroom PIR activation
should be interpreted cautiously. Recent human location, movement
between rooms, duration of bedroom presence, and other available sensor
observations should be considered.

The resident's typical routine and historical bedroom activity patterns
may provide additional contextual evidence but should not override
contradictory current sensor observations.

When the available sensor evidence supports activity in the bedroom but
does not reliably identify a more specific bedroom activity, the
activity should remain classified as bedroom personal activity rather
than inferring an unsupported specific behavior.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "bedroom_personal_activity",
    }
)

# preparing breakfast
Document(
    page_content="""
Preparing breakfast refers to food or drink preparation associated with
the resident's morning meal.

Resident presence in the kitchen provides important location evidence
for preparing breakfast. Repeated or sustained kitchen PIR activation
strengthens this evidence.

Activation of breakfast-related appliances provides strong supporting
evidence. Relevant appliances may include a kettle, toaster, microwave,
coffee maker, or other food-preparation appliances available in the
household.

The simultaneous or sequential activation of multiple breakfast-related
appliances while the resident is present in the kitchen provides stronger
evidence for preparing breakfast than activation of a single appliance.

For example, a sequence involving entering the kitchen, activating the
kettle or toaster, remaining in the kitchen, and subsequently moving
toward a dining area is consistent with preparing breakfast.

Time of day provides useful contextual evidence. Preparing breakfast should be distinguished from general cooking.
Kitchen activity and food-preparation appliance use during the resident's typical breakfast period increase the likelihood of preparing breakfast. 
However, time of day should not be used as the sole basis for classification because the resident may prepare breakfast earlier or later than usual.

Preparing breakfast should also be distinguished from eating breakfast.
Kitchen presence combined with food-preparation appliance interaction
provides stronger evidence for preparation, while movement to a dining
area followed by sustained presence and reduced appliance interaction
may provide stronger evidence for eating.

A brief kitchen PIR activation without food-preparation appliance
activity provides weak evidence for preparing breakfast because the
resident may simply be entering the kitchen, obtaining an item, or
passing through.

NILM appliance detections are probabilistic. High-confidence detections
of breakfast-related appliances should provide stronger evidence than
low-confidence detections. The absence of a detected appliance should
not completely rule out preparing breakfast because some breakfast
preparation may not require an electrical appliance.

The resident's historical breakfast routine may provide additional
context. For example, recurring morning patterns involving kitchen
presence followed by kettle and toaster activation may strengthen the
classification. Historical patterns should not override contradictory
current sensor observations.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "preparing_breakfast",
    }
)

# resting on the couch or the chairs
Document(
    page_content="""
Resting on the couch or chairs refers to a period during which the
resident is sitting or reclining in a living or seating area without
strong evidence of another more specific activity.

Resident presence in the living room or another area containing a couch
or chairs provides important location evidence for this activity.

A transition into the living or seating area followed by reduced PIR
activity is consistent with resting. PIR inactivity should not be
interpreted as evidence that the resident has left the area because a
seated or reclining person may remain sufficiently still that the PIR
sensor does not detect movement.

Occasional PIR activations separated by periods of inactivity may also
be consistent with resting because the resident may periodically change
position, stand briefly, or move while remaining in the same area.

Resting should be distinguished from simply passing through the living
room. A short isolated PIR activation provides weak evidence for
resting, while evidence that the resident entered the area and did not
subsequently trigger sensors in another location strengthens the
likelihood of continued presence and resting.

Resting should be distinguished from watching television. If the
television is active and there is evidence that the resident remains in
the room containing the television, watching television may be more
strongly supported. If the television is inactive and there is no
evidence of another specific activity, sustained presence with limited
movement may support resting.

Resting should also be distinguished from sleeping. Long periods of
very limited movement, especially in the bedroom or during the
resident's normal sleep period, may provide stronger evidence for
sleeping. Resting on the couch or chairs is more strongly supported when
the inferred location corresponds to a known seating area and the
temporal pattern is consistent with daytime or evening rest.

Electrical appliance activity is generally not required for resting.
Therefore, the absence of NILM-detected appliance activity should not
reduce the likelihood of resting when location and temporal evidence
support it.

If unrelated appliances are operating automatically, their activity
should not be interpreted as evidence against resting.

If a pet is present in the household, isolated PIR activation near a
couch or chair should be interpreted cautiously. Recent human location,
movement between rooms, duration of inferred presence, appliance
interactions, and other available sensor observations should be
considered before attributing the motion to the resident.

The resident's typical resting locations, times, and historical patterns
may provide additional contextual evidence but should not override
contradictory current sensor observations.

Evidence for resting on the couch or chairs is strongest when the
temporal sensor sequence indicates that the resident entered a known
seating area, no subsequent movement to another location was detected,
movement became limited or intermittent, and there is no stronger
evidence for another specific activity.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "resting_on_couch_or_chairs",
    }
)

# having breakfast
Document(
    page_content="""
Having breakfast refers to the period during which the resident is
consuming their morning meal after or during breakfast preparation.

Resident presence in the dining area, kitchen seating area, or another
location normally used for eating provides important location evidence
for having breakfast.

A temporal transition from preparing breakfast to sustained presence in
an eating location provides strong evidence for having breakfast. For
example, kitchen activity involving a kettle, toaster, microwave, or
other food-preparation appliance followed by movement to the dining area
is consistent with having breakfast.

Having breakfast generally involves less interaction with
food-preparation appliances than preparing breakfast. Therefore, a
reduction or termination of kettle, toaster, microwave, or other
breakfast-related appliance activity after a period of kitchen activity
may support a transition from preparing breakfast to having breakfast.

Sustained presence in the dining area with relatively limited PIR
activity may be consistent with having breakfast because the resident
may remain seated while eating. PIR inactivity should not automatically
be interpreted as absence.

Occasional PIR activations during the meal may occur when the resident
moves, stands, or briefly returns to the kitchen.

Time of day provides useful contextual evidence. Presence in a typical
eating location during the resident's usual morning breakfast period
increases the likelihood of having breakfast. However, time of day
should not be the sole basis for classification because breakfast may
occur earlier or later than usual.

Having breakfast should be distinguished from preparing breakfast.
Repeated kitchen movement and active interaction with breakfast-related
appliances provide stronger evidence for preparation, while sustained
presence in an eating location after food preparation provides stronger
evidence for having breakfast.

Having breakfast should also be distinguished from resting or sitting.
Sitting in a dining or kitchen area during the morning does not by
itself establish that the resident is eating. Evidence of preceding food
preparation, recent breakfast-related appliance activity, the resident's
usual breakfast routine, and the temporal sequence should be considered.

Current appliance activity is not required for having breakfast.
Therefore, the absence of NILM-detected appliance activity should not
reduce the likelihood of having breakfast when preceding appliance
activity and current location provide supporting evidence.

NILM detections from the preceding part of the observation window may
be more informative than appliance detections at the current moment.
For example, recent kettle and toaster activation followed by their
deactivation and sustained dining-area presence provides useful evidence
for having breakfast.

If a pet is present, isolated PIR activity in the dining area should be
interpreted cautiously. Recent human location, preceding breakfast
preparation, movement between the kitchen and dining area, and duration
of inferred presence should be considered.

The resident's historical breakfast routine may provide additional
context. Repeated historical sequences involving breakfast preparation
followed by dining-area occupancy can strengthen the inference, but
historical patterns should not override contradictory current sensor
observations.

Evidence for having breakfast is strongest when the temporal sequence
shows morning kitchen activity and breakfast-related appliance use,
followed by reduced appliance interaction and sustained presence in the
resident's usual eating location.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "having_breakfast",
    }
)

# grabing food or drinks'
Document(
    page_content="""
Grabbing food or drinks refers to a short activity in which the resident
enters the kitchen or another food-storage area to obtain food, a snack,
or a beverage without substantial food preparation.

A transition from another room into the kitchen followed by a relatively
short period of kitchen presence provides important evidence for
grabbing food or drinks.

Brief or limited kitchen PIR activation is more consistent with grabbing
food or drinks than the sustained or repeated kitchen movement commonly
associated with preparing a meal.

The activity may involve interaction with food-storage appliances or
areas such as a refrigerator, freezer, cupboard, or pantry. Refrigerator
activity occurring together with brief kitchen presence may provide
supporting evidence, but refrigerator power consumption alone should not
be treated as reliable evidence because refrigerators operate
automatically.

Grabbing a drink may involve brief use of a kettle, coffee maker, water
dispenser, or similar appliance. However, extended interaction with
multiple food-preparation appliances provides stronger evidence for meal
or drink preparation rather than simply grabbing food or drinks.

The absence of NILM-detected appliance activity should not reduce the
likelihood of grabbing food or drinks. Many instances, such as taking
fruit from a counter, retrieving a snack from a cupboard, or getting
water, may not produce a detectable electrical appliance signature.

Grabbing food or drinks should be distinguished from preparing
breakfast or other meal preparation. Sustained kitchen presence,
repeated movement, and sequential activation of appliances such as a
toaster, microwave, kettle, or other cooking appliances provide stronger
evidence for food preparation. Short kitchen occupancy with little or no
food-preparation appliance activity provides stronger evidence for
grabbing food or drinks.

Grabbing food or drinks should also be distinguished from having a meal.
If the resident remains in a dining or seating area for an extended
period after leaving the kitchen, particularly following food
preparation, eating may be more strongly supported. A brief kitchen
visit followed by return to a previous location may instead support
grabbing a snack or drink.

A temporal sequence in which the resident leaves another room, briefly
enters the kitchen, and then returns to the previous room can provide
strong evidence for grabbing food or drinks, especially when no
substantial cooking-appliance activity occurs.

Time of day can provide contextual information but should not be used as
a primary indicator. The resident may obtain food or drinks at any time
of day.

If a pet is present, isolated kitchen PIR activation should be
interpreted cautiously. Evidence of human movement from another room,
appliance interaction, duration of kitchen presence, and subsequent
movement should be considered before attributing the event to the
resident.

The resident's historical patterns may provide additional context. For
example, repeated short visits from the living room to the kitchen and
back during the evening may support the interpretation of obtaining a
snack or drink. Historical patterns should not override contradictory
current sensor observations.

Evidence for grabbing food or drinks is strongest when the temporal
sensor sequence indicates a short visit to the kitchen or food-storage
area, little or no sustained cooking-appliance activity, and subsequent
movement away from the kitchen.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "grabbing_food_or_drinks",
    }
)

# washing dishes
Document(
    page_content="""
Washing dishes refers to cleaning dishes, cookware, or utensils in the
kitchen, either manually at the sink or with assistance from a
dishwasher.

Resident presence in the kitchen provides important location evidence
for washing dishes. Sustained or repeated kitchen PIR activation may
indicate movement associated with collecting, rinsing, washing, drying,
or putting away dishes.

Washing dishes often occurs after preparing or having a meal. A temporal
sequence involving meal preparation or eating followed by movement to
the kitchen and sustained kitchen activity provides supporting evidence
for washing dishes.

Manual dishwashing may not produce any detectable NILM appliance
activity. Therefore, the absence of an active electrical appliance
should not reduce the likelihood of washing dishes when location,
temporal sequence, and other sensor observations support the activity.

If a dishwasher is present, interaction with the dishwasher may provide
supporting evidence for washing dishes. Dishwasher activation combined
with kitchen presence may indicate loading dishes and starting the
dishwasher.

Continuous dishwasher operation should not be interpreted as continuous
dishwashing activity. Once the dishwasher has been started, the resident
may leave the kitchen while the appliance continues operating
automatically.

Kitchen humidity may increase during manual dishwashing because of hot
water and steam. An increase in humidity can therefore provide
supporting evidence, particularly when combined with sustained kitchen
presence. However, humidity changes may be delayed and may also result
from cooking or other water-related activities, so humidity should not
be treated as primary evidence.

Washing dishes should be distinguished from cooking or meal preparation.
Cooking is more strongly associated with interaction with
food-preparation appliances such as the stove, oven, microwave, kettle,
or toaster. Washing dishes is more strongly supported when cooking
appliance activity has ended and sustained kitchen activity continues.

Washing dishes should also be distinguished from grabbing food or
drinks. A brief kitchen visit with little sustained movement provides
weaker evidence for washing dishes, while longer or repeated kitchen
movement after a meal provides stronger evidence.

The temporal order of events is particularly useful. A sequence such as
preparing food, eating, returning to the kitchen, sustained kitchen
movement, and then leaving the kitchen is consistent with washing dishes.

If the dishwasher is detected by the NILM model, the timing of the
detection should be considered. Dishwasher activation near the beginning
of sustained kitchen activity may support loading and starting the
dishwasher, while dishwasher operation long after the resident has left
the kitchen should not be treated as evidence that the resident is still
washing dishes.

If a pet is present, isolated kitchen PIR activation should be
interpreted cautiously. Sustained movement, recent human location,
preceding meal-related activities, appliance interactions, and duration
of kitchen occupancy provide stronger evidence that the activity is
being performed by the resident.

The resident's historical routine may provide additional context. For
example, repeated patterns of kitchen activity shortly after breakfast,
lunch, or dinner may strengthen the interpretation of washing dishes.
Historical patterns should remain supporting evidence and should not
override contradictory current sensor observations.

Evidence for washing dishes is strongest when the resident is present
in the kitchen after a meal-related activity, kitchen movement is
sustained or repeated, cooking-related appliance interaction has
decreased, and optional supporting evidence such as increased humidity
or dishwasher interaction is present.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "washing_dishes",
    }
)

# reading books
Document(
    page_content="""
Reading books refers to a period during which the resident is seated or
reclining while reading a book, magazine, newspaper, or similar written
material.

Resident presence in a location normally used for reading, such as the
living room, bedroom, study area, couch, or chair, provides contextual
evidence for reading.

Reading usually involves relatively little body movement. Therefore,
sustained presence in a room followed by reduced or intermittent PIR
activity may be consistent with reading. PIR inactivity should not be
interpreted as evidence that the resident has left the room because a
person may remain still while reading.

Occasional PIR activations may occur when the resident changes position,
turns to obtain an item, stands briefly, or moves within the room while
continuing the activity.

Reading books generally does not require a distinctive electrical
appliance. Therefore, the absence of NILM-detected appliance activity
should not reduce the likelihood of reading when location and temporal
evidence support a sedentary activity.

Reading should be distinguished from watching television. If the
television is active and the resident is inferred to remain in the room
containing the television, watching television may be more strongly
supported. If the television is inactive and the resident remains
relatively stationary in a usual reading location, reading becomes more
plausible.

Reading should also be distinguished from resting on the couch or
chairs. Both activities may involve sustained presence and limited PIR
activity, and the available sensors may not always provide enough
information to distinguish them reliably. Evidence from the resident's
routine, usual reading location, time of day, preceding activity, and
historical patterns may help distinguish reading from general resting.

Reading should be distinguished from sleeping. Very long periods of
minimal movement in the bedroom, particularly during the resident's
usual sleep period, provide stronger evidence for sleeping. Reading is
more plausible when the resident has recently entered a known reading
location and the duration and timing are consistent with the resident's
usual reading behavior.

The temporal sequence may provide useful evidence. For example, the
resident entering the living room, sitting in a usual reading area, and
then producing little or intermittent PIR activity without television
or other entertainment-appliance activity is consistent with reading.

Time of day can provide contextual evidence if the resident has a
regular reading routine, but time should not be used as the sole basis
for classification.

Temperature and humidity generally provide little direct evidence for
reading and should be treated only as environmental context.

If a pet is present, isolated PIR activation in a reading or seating
area should be interpreted cautiously. Recent human location, movement
between rooms, duration of inferred presence, appliance interactions,
and historical behavioral patterns should be considered before
attributing the motion to the resident.

The resident's historical reading patterns can provide particularly
useful supporting evidence. For example, if the resident commonly reads
in a particular chair during a specific period of the day, sustained
presence in that location with limited movement and no television
activity may strengthen the inference of reading. Historical patterns
should not override contradictory current sensor observations.

When the available sensor evidence supports a sedentary activity but
does not reliably distinguish reading from resting or another quiet
activity, confidence in the reading classification should be reduced
rather than assuming that the resident is reading.

Evidence for reading books is strongest when the resident remains in a
usual reading location with limited or intermittent movement, no stronger
evidence exists for another sedentary activity such as watching
television or sleeping, and the resident's historical routine or recent
temporal context supports reading.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "reading_books",
    }
)

# cleaning home
Document(
    page_content="""
Cleaning home refers to household cleaning activities such as tidying,
dusting, wiping surfaces, sweeping, vacuuming, or organizing different
areas of the home.

Cleaning home is generally associated with active movement. Repeated PIR
activations within a room or across multiple rooms provide important
evidence for this activity.

Movement between several areas of the home over a relatively short
period may strengthen the evidence for cleaning, particularly when the
resident repeatedly returns to the same rooms rather than following a
simple path from one location to another.

Cleaning may involve sustained activity within a single room. Therefore,
remaining in one location with frequent or repeated PIR activations can
also be consistent with cleaning that room.

The temporal pattern of movement is important. Sequences involving
repeated room transitions, sustained motion, and revisiting previously
occupied rooms may be more consistent with cleaning than with ordinary
movement through the home.

Cleaning home does not necessarily require an electrical appliance.
Activities such as dusting, wiping surfaces, organizing objects, or
sweeping may produce no detectable NILM appliance activity. Therefore,
the absence of appliance activity should not reduce the likelihood of
cleaning when movement patterns provide strong evidence.

If electrical cleaning appliances such as a vacuum cleaner are available
in the household, their activation can provide strong supporting
evidence. High-confidence NILM detection of a vacuum cleaner combined
with repeated resident movement provides stronger evidence for cleaning
than either observation alone.

Electrical cleaning appliance activity should not automatically imply
that the resident is currently cleaning. The appliance detection should
be interpreted together with resident location, PIR activity, duration,
and the temporal sequence of events.

Cleaning home should be distinguished from ordinary movement between
rooms. A small number of sequential PIR activations representing movement
from one destination to another provides weaker evidence for cleaning
than repeated movement, room revisits, or sustained activity.

Cleaning should also be distinguished from cooking or washing dishes.
Sustained activity concentrated in the kitchen together with
food-preparation appliances provides stronger evidence for cooking.
Kitchen activity following a meal, particularly with dishwasher
interaction or increased humidity, may provide stronger evidence for
washing dishes.

Cleaning should also be distinguished from laundry. Washing-machine or
dryer interaction combined with movement to and from a laundry area may
provide stronger evidence for laundry, while broader movement across
multiple rooms may provide stronger evidence for general home cleaning.

Cleaning may cause temporary changes in temperature or humidity,
particularly when water is used or windows are opened. However,
temperature and humidity should normally be treated as weak contextual
evidence rather than primary indicators.

Duration can provide useful evidence. Cleaning commonly continues for
longer than a brief room transition and may produce repeated movement
patterns throughout the observation window.

If a pet is present, individual PIR activations or short movement
sequences should be interpreted cautiously. A coordinated pattern of
human movement across rooms, sustained activity, appliance interaction,
and recent human location provides stronger evidence that the observed
movement represents cleaning by the resident.

The resident's historical cleaning routine may provide additional
context. For example, recurring periods of sustained multi-room activity
at particular times or on particular days may strengthen the
classification. Historical patterns should not override contradictory
current sensor observations.

When sensor evidence indicates unusually high household movement but
does not clearly support a more specific activity, cleaning home may be
considered plausible. However, confidence should be reduced when the
available observations cannot reliably distinguish cleaning from other
active household behaviors.

Evidence for cleaning home is strongest when the temporal sensor sequence
shows sustained or repeated movement, frequent PIR activations, movement
within or across multiple rooms, possible revisiting of rooms, and,
when applicable, supporting activity from electrical cleaning
appliances.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "cleaning_home",
    }
)
# taking a shower
Document(
    page_content="""
Taking a shower refers to the resident entering the bathroom and using
the shower for personal hygiene.

Resident presence in the bathroom provides important location evidence
for taking a shower. A transition from another area of the home into the
bathroom followed by sustained bathroom occupancy strengthens this
evidence.

Taking a shower usually involves remaining in the bathroom for a period
of time rather than producing only a brief bathroom PIR activation.
Therefore, sustained bathroom presence provides stronger evidence for
taking a shower than a short visit.

PIR activity during a shower may be intermittent or limited. The
resident may remain within the shower area without generating frequent
motion detections. Therefore, reduced PIR activity after the resident
enters the bathroom should not be interpreted as evidence that the
resident has left.

An increase in bathroom humidity provides important supporting evidence
for taking a shower. Warm water and steam may cause humidity to rise
during or shortly after the shower.

Humidity changes may be delayed relative to the start of the activity.
Therefore, the temporal trend in humidity across the observation window
may be more informative than the humidity value at a single instant.

A substantial or sustained increase in bathroom humidity combined with
bathroom occupancy provides stronger evidence for taking a shower than
either observation alone. However, humidity should not be treated as
definitive evidence because other water-related activities or
environmental conditions may also affect humidity.

Bathroom temperature may change during or after showering and can
provide additional contextual evidence. Temperature should normally be
treated as supporting rather than primary evidence.

Taking a shower generally does not require a distinctive
NILM-detectable appliance. Therefore, the absence of appliance activity
should not reduce the likelihood of taking a shower when bathroom
occupancy and environmental evidence support it.

If the household contains electrically powered water-heating or
shower-related equipment that can be detected reliably by NILM, its
activation may provide additional supporting evidence. Such detections
should be interpreted together with bathroom occupancy and humidity
rather than used independently.

Taking a shower should be distinguished from general bathroom use.
Brief bathroom occupancy without a meaningful humidity increase provides
stronger evidence for ordinary bathroom use. Longer bathroom occupancy
combined with a rising humidity pattern provides stronger evidence for
taking a shower.

Taking a shower should also be distinguished from other personal
activities performed in the bathroom. Bathroom location alone is not
sufficient to identify showering. Duration, humidity changes,
temperature changes, and the temporal sequence of sensor observations
should be considered together.

The temporal sequence provides particularly useful evidence. A pattern
in which the resident enters the bathroom, remains there for several
minutes, bathroom humidity begins to increase, and the resident later
leaves the bathroom is consistent with taking a shower.

Humidity may remain elevated after the resident has finished showering
and left the bathroom. Therefore, elevated humidity after bathroom exit
should not be interpreted as evidence that the shower activity is still
continuing.

If a pet is present, isolated bathroom PIR activation should be
interpreted cautiously. Sustained bathroom occupancy together with a
humidity increase and recent human movement into the bathroom provides
much stronger evidence that the activity was performed by the resident.

Time of day and the resident's historical shower routine may provide
additional contextual evidence. For example, a recurring pattern of
bathroom occupancy and increasing humidity during the resident's usual
shower period may strengthen the inference. Historical patterns should
not override contradictory current sensor observations.

When bathroom occupancy is detected but humidity or other supporting
evidence is unavailable, taking a shower may remain plausible but the
confidence should be reduced rather than assuming that the resident is
showering.

Evidence for taking a shower is strongest when the temporal sensor
sequence shows movement into the bathroom, sustained bathroom occupancy,
a rising bathroom humidity pattern, possible supporting temperature
changes, and subsequent movement out of the bathroom.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "taking_a_shower",
    }
)

# playing video games
Document(
    page_content="""
Playing video games refers to a period during which the resident is
actively engaged with a video game using a gaming console, computer,
television, or another gaming device.

Resident presence in a location where gaming equipment is available,
such as the living room, bedroom, or entertainment area, provides
important location evidence for playing video games.

Playing video games is generally a sedentary activity and may involve
relatively little whole-body movement. Therefore, sustained presence in
a gaming location followed by limited or intermittent PIR activity can
be consistent with playing video games. PIR inactivity should not be
interpreted as evidence that the resident has left the area.

Occasional PIR activations may occur when the resident changes position,
stands briefly, retrieves an item, or moves within the room while
continuing the gaming session.

Activation of a known gaming console or gaming computer provides strong
supporting evidence for playing video games when the resident is also
inferred to be present in the same room.

If gaming involves a television, simultaneous television and gaming
device activity can provide stronger evidence for playing video games
than television activity alone.

NILM appliance detections are probabilistic. High-confidence detection
of a gaming console or other gaming device like PCs should provide stronger
evidence than a low-confidence detection. Appliance evidence should
always be interpreted together with resident location and the temporal
sensor sequence.

Playing video games should be distinguished from watching television.
Television activity combined with sedentary living-room presence may
support either activity. Detection of an active gaming console,
computer, or other gaming device provides stronger evidence for playing
video games, while television activity without evidence of a gaming
device may provide stronger evidence for watching television.

Playing video games should also be distinguished from resting on the
couch or chairs. Both activities may involve sustained presence and
limited PIR activity. Active gaming equipment provides useful evidence
for gaming, while the absence of relevant device activity and the
absence of other specific evidence may support general resting.

Playing video games may also be difficult to distinguish from computer
use when a computer is used for gaming. If the available sensors cannot
determine how the computer is being used, the evidence may be
insufficient to confidently distinguish gaming from other computer-based
activities. In such cases, classification confidence should be reduced.

The duration of gaming sessions can vary considerably. Sustained
presence in the gaming location together with continuous or repeated
gaming-device activity across the observation window strengthens the
evidence for playing video games.

A gaming device remaining powered on does not necessarily mean that the
resident is actively playing. If the resident leaves the room while the
device remains active, appliance activity alone should not be interpreted
as continued gaming.

Similarly, standby power or background electricity consumption from a
gaming console should not be interpreted as evidence of playing video
games. Resident presence and meaningful device activation should be
considered together.

Temperature and humidity generally provide little direct evidence for
playing video games and should be treated only as environmental context.

If a pet is present, isolated PIR activation near the entertainment area
should be interpreted cautiously. Recent human location, duration of
presence, gaming-device activity, and movement between rooms should be
considered before attributing the activity to the resident.

The resident's historical gaming patterns may provide additional
context. For example, repeated periods of gaming-device activity and
living-room or bedroom presence during particular times may strengthen
the inference. Historical patterns should not override contradictory
current sensor observations.

When the sensor evidence supports a sedentary entertainment activity but
cannot reliably distinguish playing video games from watching television,
computer use, or resting, confidence in the gaming classification should
be reduced rather than assuming a specific activity.

Evidence for playing video games is strongest when the resident remains
in a location containing gaming equipment, PIR activity is consistent
with a sedentary activity, a gaming console or other gaming device is
actively detected, and the temporal sequence indicates that the resident
remains present while the gaming equipment is in use.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "playing_video_games",
    }
)

# getting out of the apartment
Document(
    page_content="""
Getting out of home refers to the resident leaving the monitored
residence through an entrance or exit door.

The residence may be an apartment, condominium unit, detached house,
semi-detached house, townhouse, or another type of dwelling. The
activity should therefore be inferred relative to the monitored
residential boundary rather than assuming a particular building type.

The entrance or exit door sensor provides important evidence for
detecting this activity. A door-opening event indicates that someone may
be entering or leaving the residence, but a door event alone is not
sufficient to determine the direction of movement.

The temporal sequence of indoor PIR activity and the entrance-door event
is particularly important for determining whether the resident is
leaving the home.

A sequence in which the resident is detected inside the home, moves
toward the entrance or exit area, the entrance door opens, and indoor
resident activity subsequently disappears provides strong evidence for
getting out of home.

For example, PIR activation in an interior room followed by hallway or
entrance-area PIR activation and then an entrance-door opening is
consistent with movement toward the exit. If no further human indoor
activity is detected afterward, the likelihood that the resident left
the residence increases.

If an entrance-area PIR sensor is available, its activation immediately
before the door opens can provide useful evidence that the resident
approached the exit from inside the residence.

The entrance door may generate both opening and closing events during
departure. A sequence involving movement toward the entrance, door
opening, possible door closing, and subsequent absence of indoor human
activity provides stronger evidence for leaving than a single door
event.

Getting out of home should be distinguished from getting into home.
For leaving, indoor movement generally occurs before the entrance-door
event and is followed by reduced or absent indoor resident activity.
For entering, the entrance-door event is generally followed by indoor
PIR activations indicating movement from the entrance into the
residence.

A door-opening event without a corresponding change in resident location
should not automatically be classified as getting out of home. The
resident may open the door to receive a delivery, speak with someone,
retrieve mail, dispose of garbage, check outside, or perform another
brief activity without leaving the monitored residence for a meaningful
period.

A brief exit followed quickly by another entrance-door event and renewed
indoor PIR activity may indicate a short temporary departure. Depending
on the activity definitions used by the recognition system, this may
still be classified as getting out of home, but the duration outside
should be considered when interpreting the event.

The absence of PIR activity immediately after a door event should be
interpreted using an appropriate temporal window. PIR sensors may not
detect a stationary resident, so a very short period without PIR
activation is not sufficient by itself to confirm that the resident has
left.

Evidence from multiple indoor sensors can strengthen the inference. If
no PIR sensor associated with the resident is activated for a sustained
period after the entrance-door event, this provides stronger evidence
that the resident is outside the monitored residence.

Electrical appliance activity is generally not required for getting out
of home. Appliances may remain active after the resident leaves, so
continued NILM-detected appliance activity should not automatically be
interpreted as evidence that the resident remains at home.

Similarly, automatically operating appliances such as refrigerators,
heating systems, washing machines, dishwashers, or other unattended
devices may continue operating while the resident is away.

If the resident switches off or stops interacting with personal or
entertainment appliances shortly before moving toward the entrance, this
may provide additional contextual evidence for departure but should not
be required.

If a pet is present in the household, indoor PIR activity after the
resident leaves may be generated by the pet. Therefore, continued PIR
activity after an entrance-door event should not automatically be
interpreted as evidence that the resident remains at home.

When pets are present, the system should consider the spatial and
temporal movement pattern, recent confirmed human location, sensor
characteristics, and other available evidence when distinguishing human
activity from pet-generated PIR events.

For a single-resident household, a confirmed departure has an important
effect on subsequent activity interpretation. Once the resident is
inferred to have left the residence, subsequent indoor sensor events
should not automatically be attributed to the resident until evidence
of returning home is observed.

For apartments, leaving the monitored home means crossing the entrance
door of the apartment unit. Activity in a shared hallway, lobby,
elevator, stairwell, parking area, or other common building space is
outside the monitored residence unless those areas are explicitly part
of the sensor system.

For houses, leaving the monitored home generally means crossing an
external entrance or exit door. If multiple exterior doors exist, such
as front, back, side, patio, or garage-entry doors, any door that crosses
the defined monitored residential boundary may provide evidence for
leaving.

If multiple entrance or exit doors exist, metadata describing each
door's role and location should be used rather than assuming that only
the front door can be used for departure.

Time of day and the resident's historical departure routine may provide
additional contextual evidence. For example, regularly leaving home at
a particular time may strengthen an otherwise consistent departure
sequence. Historical routines should not override contradictory current
sensor observations.

When a door event occurs but the direction of movement cannot be
determined reliably from surrounding sensor observations, the system
should maintain uncertainty rather than automatically classifying the
event as getting out of home.

Evidence for getting out of home is strongest when the temporal sensor
sequence shows confirmed resident presence inside the home, movement
toward an entrance or exit, activation of the corresponding door sensor,
and a subsequent sustained absence of indoor activity attributable to
the resident.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "getting_out_of_home",
    }
)

# coming into the apartment
Document(
    page_content="""
Coming into the apartment refers to the resident entering the monitored
residence from outside through the main entrance or another designated
entrance door.

Although the activity label refers to an apartment, the reasoning can
also be generalized to other residential settings when the monitored
boundary and entrance doors are clearly defined.

The entrance-door sensor provides important evidence for detecting this
activity. A door-opening event indicates that someone may be entering or
leaving the residence, but the door event alone is not sufficient to
determine the direction of movement.

The temporal sequence of the entrance-door event and subsequent indoor
PIR activations is particularly important for identifying entry into the
apartment.

A sequence in which the entrance door opens and is followed by PIR
activation in the entrance, hallway, or another interior area provides
strong evidence that someone has entered the residence.

Evidence becomes stronger when subsequent PIR activations indicate
movement from the entrance area toward interior rooms such as the living
room, kitchen, bedroom, or other areas of the apartment.

For example, a sequence involving entrance-door opening, entrance or
hallway PIR activation, and then living-room or kitchen PIR activation
is consistent with the resident coming into the apartment.

If the entrance door produces both opening and closing events, a sequence
of door opening, indoor PIR activation, door closing, and continued
movement into the residence provides particularly strong evidence for
entry.

Coming into the apartment should be distinguished from getting out of
home. For entry, the entrance-door event is generally followed by indoor
movement away from the entrance and toward interior areas. For departure,
indoor movement generally occurs before the entrance-door event and is
followed by a sustained absence of resident-attributable indoor activity.

A door-opening event without subsequent evidence of movement into the
residence should not automatically be classified as coming into the
apartment. The door may have been opened by someone already inside, by a
visitor, for a delivery, or for another brief interaction at the
entrance.

In a single-resident household, prior evidence that the resident was
outside the home substantially strengthens the interpretation of a new
entrance-door event followed by indoor movement as the resident returning
home.

The system should maintain an estimate of the resident's previous
home-presence state when possible. A transition from an inferred
outside-home state to entrance-door activation followed by indoor PIR
activity provides stronger evidence for coming into the apartment than
the door event alone.

The absence of PIR activity immediately after the entrance-door event
does not completely rule out entry. PIR sensors may have limited
coverage, and the resident may pause near the entrance before moving
farther inside. Evidence should therefore be evaluated over an
appropriate temporal window.

Electrical appliance activity is not required for recognizing entry into
the apartment. Therefore, the absence of NILM-detected appliance activity
should not reduce the likelihood of this activity when door and movement
evidence strongly support it.

Appliance interaction shortly after entry may provide additional
supporting context. For example, the resident may turn on lights,
television, kettle, or another appliance after returning home. Such
activity may strengthen evidence of occupancy but should not be required
for identifying entry.

If appliances were already operating while the resident was outside,
their continued operation should not be interpreted as evidence that the
resident was already inside.

If a pet is present, indoor PIR activity may occur before, during, and
after the resident enters the apartment. Therefore, PIR activation alone
should not be used to identify the resident's return.

When a pet is present, the combination of an entrance-door event with a
spatial sequence of PIR activations moving from the entrance toward the
interior provides stronger evidence of human entry than isolated indoor
PIR events.

For an apartment, the monitored residential boundary normally begins at
the apartment-unit entrance door. Shared hallways, elevators, stairwells,
lobbies, parking areas, and other common building spaces should normally
be considered outside the apartment unless they are explicitly included
in the monitoring system.

If the residence has more than one possible entrance, metadata describing
the location and role of each entrance door should be considered. The
reasoning should not assume that the resident always enters through a
single specific door.

Time of day and the resident's historical return-home routine may provide
additional contextual evidence. For example, recurring return times may
strengthen an otherwise consistent entry sequence. Historical routines
should not override contradictory current sensor observations.

When an entrance-door event occurs but the direction of movement cannot
be determined from the surrounding sensor observations, the system should
maintain uncertainty rather than automatically assuming that the resident
has entered.

Evidence for coming into the apartment is strongest when the resident was
previously inferred to be outside, the entrance door opens, indoor PIR
activity begins near the entrance, and subsequent sensor activations show
movement from the entrance toward the interior of the residence.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "coming_into_home",
    }
)

# chatting/using the cell-phone
Document(
    page_content="""
Chatting or using the cell phone refers to a period during which the
resident is communicating with family members or other people, either
through an in-person conversation or by using a cell phone for a voice
call or similar communication.

This activity may occur in different locations within the home, such as
the living room, kitchen, dining area, bedroom, or another occupied
area. Therefore, no single room should be treated as the required
location for this activity.

Chatting or using the cell phone often produces limited direct evidence
from PIR, environmental, door, and electrical sensors. The activity
should therefore be inferred cautiously and should not be assigned with
high confidence when the available sensors only indicate general
presence or inactivity.

During a phone conversation, the resident may remain seated or standing
in one location for an extended period. PIR activity may therefore be
limited or intermittent. A lack of frequent PIR activation should not
be interpreted as evidence that the resident has left the area.

The resident may also move between rooms while talking on the phone.
Repeated movement without interaction with household appliances may be
consistent with a mobile phone conversation, but this pattern alone is
not sufficient to identify the activity.

In-person chatting may involve one or more other people being present in
the residence. However, standard PIR sensors generally cannot reliably
determine the number or identity of people in a room. Therefore, PIR
activity alone should not be used to conclude that an in-person
conversation is occurring.

If household context indicates that a family member or visitor is
currently present, sustained resident presence in the same area may
provide contextual support for in-person chatting. Visitor or household
presence should be based on available evidence or known context rather
than assumed by the activity-recognition system.

Using a cell phone generally does not create a distinctive signature in
aggregate electricity measurements. Therefore, the absence of
NILM-detected appliance activity should not reduce the likelihood of
phone use.

Similarly, household appliance activity should not normally be used as
primary evidence for this activity unless it helps distinguish another
more specific behavior.

Chatting or using the cell phone should be distinguished from watching
television. Television activity together with resident presence in the
television area provides stronger evidence for watching television. If
the television is inactive and the resident remains sedentary, chatting
or phone use may remain plausible, but the absence of television
activity alone is not sufficient to identify it.

Chatting or using the cell phone should also be distinguished from
reading books and resting on the couch or chairs. All of these
activities may involve sustained presence with little PIR activity.
Without additional contextual evidence, the available sensors may not
reliably distinguish among these sedentary activities.

The resident's recent activity sequence can provide useful context. For
example, the resident may stop another activity, remain in one location
for a period with limited movement, and later resume the previous
activity. Such a sequence may be compatible with a conversation or phone
call but should remain uncertain without stronger supporting evidence.

If information about phone usage or call events is explicitly available
to the system from an authorized data source, a call occurring while the
resident is inferred to be present can provide strong supporting
evidence for phone-based chatting. Such information should only be used
when it is actually available and should never be inferred from unrelated
sensor observations.

Time of day may provide contextual information if the resident regularly
communicates with family members at particular times. However, time of
day should not be used as the sole basis for identifying chatting or
cell-phone use.

The resident's historical communication patterns may provide useful
personal context. For example, if the resident regularly speaks with a
family member at a particular time, a compatible period of sedentary
activity may increase the plausibility of chatting or phone use.
Historical patterns should remain supporting evidence and should not
override contradictory current sensor observations.

Temperature and humidity generally provide little or no direct evidence
for chatting or using the cell phone and should be treated only as
environmental context.

If a pet is present, isolated PIR activity should be interpreted
cautiously and should not be used as evidence of conversation or phone
use.

When the available sensor observations are equally compatible with
chatting, reading, resting, or another sedentary activity, the system
should maintain uncertainty and reduce classification confidence rather
than inventing evidence for communication.

Evidence for chatting or using the cell phone is strongest when
additional contextual information supports communication, such as known
presence of a family member or visitor, an explicitly available phone
call event, or a strong resident-specific communication pattern combined
with compatible location and temporal observations.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "chatting_or_using_cell_phone",
    }
)

# having fruits or snacks
Document(
    page_content="""
Having fruits or snacks refers to the period during which the resident
is consuming fruit, a snack, or another small food item outside or
between regular meals.

This activity may occur in the kitchen, dining area, living room, or
another location where the resident commonly sits or eats. Therefore,
no single room should be treated as a required location for having
fruits or snacks.

Having fruits or snacks often follows a short visit to the kitchen or
another food-storage area. A temporal sequence in which the resident
briefly enters the kitchen, obtains an item, leaves the kitchen, and
then remains in a dining or seating area is consistent with having
fruits or snacks.

The distinction between grabbing food or drinks and having fruits or
snacks is particularly important. Grabbing food or drinks refers
primarily to the short retrieval activity, while having fruits or snacks
refers to consuming the retrieved food.

A brief kitchen visit followed quickly by movement away from the kitchen
provides stronger evidence for grabbing food or drinks. If this sequence
is followed by sustained presence in a dining area, living room, or
other usual eating location, the later portion of the sequence provides
stronger evidence for having fruits or snacks.

The activity may also occur entirely in the kitchen. In this case, the
transition from short food-retrieval behavior to a period of relatively
stationary kitchen presence may support having fruits or snacks rather
than continued grabbing behavior.

Having fruits or snacks usually requires less food preparation than
preparing breakfast, lunch, dinner, or another meal. Therefore, the
absence of sustained cooking activity or multiple food-preparation
appliance activations may support the interpretation of a snack rather
than meal preparation.

Electrical appliance activity is not required for having fruits or
snacks. Eating fruit, packaged snacks, bread, nuts, or other ready-to-eat
foods may produce no detectable NILM appliance activity. Therefore, the
absence of appliance activity should not reduce the likelihood of this
activity when temporal and location evidence support it.

Recent refrigerator interaction may provide supporting context if the
resident retrieves fruit, refrigerated food, or a drink. However,
refrigerator electricity consumption alone should not be interpreted as
evidence of either grabbing or consuming food because refrigerators
operate automatically.

Occasional use of a microwave, kettle, toaster, or another appliance may
occur when preparing a snack. However, sustained interaction with
multiple food-preparation appliances and prolonged kitchen activity may
provide stronger evidence for meal preparation rather than having a
simple snack.

Having fruits or snacks should be distinguished from grabbing food or
drinks based primarily on activity phase and temporal sequence.
Movement toward food storage and short kitchen occupancy characterize
the retrieval phase, while subsequent sustained presence with reduced
movement is more consistent with the consumption phase.

For example, a sequence involving living-room presence, a short kitchen
visit, and return to the living room is initially consistent with
grabbing food or drinks. If the resident then remains in the living
room with relatively limited movement, the subsequent period may be
consistent with having fruits or snacks.

Having fruits or snacks should also be distinguished from drinking.
When available sensor evidence indicates only that the resident
retrieved an item from the kitchen, it may not be possible to determine
whether the item was food or a beverage. The system should not infer
fruit or snack consumption unless additional temporal, contextual, or
historical evidence supports it.

Having fruits or snacks should be distinguished from having breakfast or
another main meal. Main meals are generally associated with the
resident's meal routine, longer eating periods, and often preceding meal
preparation. Snack consumption may be shorter and can occur between
regular meal periods, although time of day should not be used as a
strict rule.

PIR activity during snack consumption may be limited because the
resident may remain seated while eating. Therefore, low or absent PIR
activity after the resident reaches an eating or seating location should
not be interpreted as evidence that the resident has left.

Duration can provide useful contextual evidence. The retrieval stage is
typically brief, while consuming a snack or fruit may involve a longer
period of relatively stationary presence. Duration alone, however,
should not determine the activity.

Temperature and humidity generally provide little direct evidence for
having fruits or snacks and should be treated only as environmental
context.

If a pet is present, isolated kitchen, dining-room, or living-room PIR
activations should be interpreted cautiously. Recent confirmed human
location, movement between rooms, duration of presence, and the temporal
sequence of events should be considered before attributing the activity
to the resident.

The resident's historical snack patterns may provide additional
context. For example, a recurring pattern involving a short kitchen
visit followed by returning to a preferred chair or couch during the
afternoon may strengthen the interpretation of having fruits or snacks.
Historical patterns should not override contradictory current sensor
observations.

When the sensor evidence supports food or drink retrieval but does not
provide sufficient information to determine whether the resident
subsequently consumed food, the system should maintain uncertainty
rather than automatically classifying the activity as having fruits or
snacks.

Evidence for having fruits or snacks is strongest when the temporal
sensor sequence indicates retrieval of food from the kitchen or another
food-storage area followed by sustained presence in a usual eating or
seating location, reduced movement consistent with eating, little or no
substantial meal-preparation activity, and contextual or historical
evidence consistent with snack consumption.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "having_fruits_or_snacks",
    }
)

# using_pc
Document(
    page_content="""
Using PC refers to a period during which the resident is actively using
a desktop computer, laptop, or similar personal computer for general
activities such as browsing the internet, reading, writing, email,
communication, administrative tasks, work, study, or other non-gaming
computer use.

Resident presence in a location where a computer is normally used, such
as a study, bedroom, office area, desk, or living room, provides
important location evidence for using a PC.

Using a PC is generally a sedentary activity and may involve very little
whole-body movement. Therefore, sustained presence near the computer
with limited or intermittent PIR activity can be consistent with PC use.
A lack of frequent PIR activation should not be interpreted as evidence
that the resident has left the area.

Occasional PIR activation may occur when the resident changes position,
stands briefly, retrieves an item, or moves around the desk or computer
area.

If a desktop computer, monitor, or related computer equipment can be
identified by NILM, activation of this equipment while the resident is
present in the corresponding location provides useful supporting
evidence for using a PC.

For laptops and other battery-powered devices, NILM evidence may be weak
or completely absent. Therefore, the absence of detectable computer
electricity consumption should not rule out PC use.

Computer equipment may remain powered on when it is not actively being
used. Consequently, computer or monitor electricity consumption alone
should not be interpreted as proof that the resident is using the PC.
Resident presence and the temporal sequence of activity should also be
considered.

Using a PC should be carefully distinguished from playing video games.
Both activities may involve the same computer, the same location,
similar durations, and similarly low levels of PIR activity.

Detection of computer activity alone provides evidence for PC use but
does not reveal whether the resident is working, browsing, communicating,
reading, or playing a video game.

Playing video games should be preferred only when additional
gaming-specific evidence is available. Such evidence may include
activation of a dedicated gaming console, known gaming equipment, or
other contextual or historical information indicating that the computer
is being used for gaming.

If a PC is used for both gaming and general computer activities and no
gaming-specific evidence is available, the system should not infer
gaming solely from PC activation. In this situation, the broader
using-PC label is better supported by the observable evidence.

If a dedicated gaming console is active together with a television and
the resident is present in the corresponding entertainment area, this
provides stronger evidence for playing video games than for general PC
use.

Using a PC should also be distinguished from watching television.
Television activity combined with resident presence in the television
area provides stronger evidence for watching television, while computer
or monitor activity combined with presence near the usual computer
location provides stronger evidence for using a PC.

Using a PC should be distinguished from reading books and resting.
These activities may all produce prolonged periods of low PIR activity.
Computer-equipment activation and presence in the resident's usual
computer location provide additional evidence for PC use. When such
evidence is unavailable, the classification should remain uncertain.

The temporal sequence can provide useful evidence. A pattern in which
the resident enters the room containing the computer, computer or
monitor activity begins, the resident remains in that location with
limited movement, and the computer activity later ends is consistent
with using a PC.

A computer remaining active after the resident moves to another room
should not be interpreted as continued PC use. Location evidence should
therefore be considered together with device activity.

Duration alone should not distinguish PC use from gaming. Both
activities may last for short or extended periods.

Time of day may provide contextual information if the resident has
regular periods for computer use or gaming. However, time of day should
not be used as the primary distinction between these activities.

Temperature and humidity generally provide little direct evidence for
PC use and should be treated only as environmental context.

If a pet is present, isolated PIR activity near the computer area should
be interpreted cautiously. Recent human location, sustained presence,
computer-equipment activity, and movement between rooms should be
considered before attributing PC use to the resident.

The resident's historical computer-use patterns may provide additional
context. For example, repeated periods of computer activation and
presence at a desk during particular times may strengthen the inference
of using a PC. Historical patterns should not override contradictory
current sensor observations.

When the sensor evidence confirms computer use but cannot determine
whether the resident is performing a general PC activity or playing a
computer game, the system should avoid confidently assigning the gaming
label without gaming-specific evidence.

Evidence for using a PC is strongest when the resident remains in the
usual computer location, PIR activity is consistent with a sedentary
activity, computer or monitor activity is detected when available, and
there is no stronger gaming-specific evidence indicating that the
computer is being used for playing video games.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "using_pc",
    }
)

# preparing dinner
Document(
    page_content="""
Preparing dinner refers to food or drink preparation associated with
the resident's evening meal.

Resident presence in the kitchen provides important location evidence
for preparing dinner. Sustained or repeated kitchen PIR activation
strengthens this evidence because dinner preparation commonly involves
movement between food-storage, preparation, cooking, and serving areas.

Activation of food-preparation and cooking appliances provides strong
supporting evidence. Relevant appliances may include the oven, stove,
microwave, kettle, toaster, air fryer, or other cooking appliances
available in the household.

The simultaneous or sequential activation of multiple cooking-related
appliances while the resident is present in the kitchen provides
stronger evidence for preparing dinner than activation of a single
appliance.

A temporal sequence involving entering the kitchen, sustained kitchen
movement, interaction with one or more cooking appliances, and
subsequent movement toward a dining or eating area is consistent with
preparing dinner.

Dinner preparation may involve periods when an appliance remains active
while the resident performs other tasks or briefly leaves the kitchen.
Therefore, continuous appliance activity does not necessarily imply
continuous direct interaction with that appliance.

Similarly, some dinner preparation may not require detectable electrical
appliance activity. Preparing a salad, sandwich, cold meal, or other
uncooked food may produce little or no NILM-detectable appliance
activity. The absence of cooking-appliance activity should therefore not
completely rule out preparing dinner when location, movement, timing,
and temporal context support it.

NILM appliance detections are probabilistic. High-confidence detections
of cooking-related appliances should provide stronger evidence than
low-confidence detections. Appliance detections should be interpreted
together with resident location and the temporal sequence rather than
used independently.

Time of day provides useful contextual evidence for distinguishing
preparing dinner from other meal-preparation activities. Kitchen
activity and cooking-appliance use during the resident's typical evening
meal-preparation period increase the likelihood of preparing dinner.

Time of day should not be used as the sole basis for classification.
The resident may prepare dinner earlier or later than usual, and similar
cooking behavior may occur at other times of day.

Preparing dinner should be distinguished from preparing breakfast.
Both activities may involve sustained kitchen presence and appliance
interaction. Evening timing, the resident's usual dinner routine, and
the types and duration of cooking-appliance activity may provide
supporting evidence for dinner rather than breakfast preparation.

Preparing dinner should be distinguished from grabbing food or drinks.
A short kitchen visit with little or no sustained food-preparation
activity provides stronger evidence for grabbing food or drinks.
Longer kitchen occupancy, repeated movement, and cooking-appliance
interaction provide stronger evidence for preparing dinner.

Preparing dinner should also be distinguished from having fruits or
snacks. Snack consumption is more likely to involve brief food
retrieval followed by relatively stationary presence in an eating or
seating location. Sustained kitchen activity and cooking-related
appliance use provide stronger evidence for dinner preparation.

Preparing dinner should be distinguished from washing dishes. Cooking
appliance interaction during sustained kitchen activity provides
stronger evidence for preparing dinner. If cooking activity has ended
and sustained kitchen movement continues, particularly after eating,
washing dishes may become more strongly supported.

The transition from preparing dinner to having dinner is also important.
Kitchen presence and active food preparation provide stronger evidence
for preparing dinner, while movement toward a dining or usual eating
area followed by sustained presence and reduced cooking-appliance
interaction may indicate that dinner preparation has ended and eating
has begun.

Temperature and humidity changes in the kitchen may provide additional
supporting evidence. Cooking with an oven, stove, boiling water, or
other heat-producing processes may increase temperature or humidity.
However, environmental changes may be delayed and should not be treated
as primary evidence.

If a pet is present, isolated kitchen PIR activation should be
interpreted cautiously. Sustained human movement, recent confirmed human
location, cooking-appliance interaction, duration of kitchen occupancy,
and the temporal sequence provide stronger evidence that the resident is
preparing dinner.

The resident's historical dinner routine may provide additional context.
For example, recurring evening patterns involving sustained kitchen
presence followed by cooking-appliance activation and movement toward
the dining area may strengthen the classification. Historical patterns
should not override contradictory current sensor observations.

When kitchen activity is detected but the available evidence cannot
reliably determine whether the resident is preparing dinner, preparing
another meal, obtaining a snack, or performing another kitchen activity,
the system should maintain uncertainty and reduce classification
confidence rather than relying only on time of day.

Evidence for preparing dinner is strongest when evening contextual
timing, sustained or repeated kitchen presence, food-preparation or
cooking-appliance activity, and the temporal sequence of sensor events
consistently support preparation of the resident's evening meal.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "preparing_dinner",
    }
)

# preparing lunch
Document(
    page_content="""
Preparing lunch refers to food or drink preparation associated with the
resident's midday meal.

Resident presence in the kitchen provides important location evidence
for preparing lunch. Sustained or repeated kitchen PIR activation
strengthens this evidence because lunch preparation may involve movement
between food-storage, preparation, cooking, and serving areas.

Activation of food-preparation or cooking appliances provides useful
supporting evidence. Relevant appliances may include the microwave,
stove, oven, kettle, toaster, air fryer, or other appliances available
in the household.

The simultaneous or sequential activation of multiple food-preparation
appliances while the resident is present in the kitchen provides
stronger evidence for preparing lunch than activation of a single
appliance.

A temporal sequence involving entering the kitchen, sustained kitchen
movement, interaction with food or cooking appliances, and subsequent
movement toward a dining or eating area is consistent with preparing
lunch.

Lunch preparation may also involve little or no detectable electrical
appliance activity. For example, preparing a sandwich, salad, cold meal,
or other ready-to-eat food may not produce a distinctive NILM signature.
Therefore, the absence of appliance activity should not rule out
preparing lunch when location, movement, timing, and temporal context
support it.

NILM appliance detections are probabilistic. High-confidence detections
of food-preparation appliances should provide stronger evidence than
low-confidence detections. Appliance evidence should be interpreted
together with resident location and the temporal sequence rather than
used independently.

Time of day provides useful contextual evidence for distinguishing
preparing lunch from other meal-preparation activities. Kitchen activity
during the resident's typical midday meal-preparation period increases
the likelihood of preparing lunch.

Time of day should not be used as the sole basis for classification.
The resident may prepare lunch earlier or later than usual, and similar
food-preparation behavior may occur at other times of day.

Preparing lunch should be distinguished from preparing breakfast and
preparing dinner. These activities may produce very similar sensor
patterns because they all involve kitchen presence and food preparation.
The resident's usual meal schedule, recent activity sequence, appliance
usage patterns, and time of day may provide contextual evidence for
distinguishing among them.

Preparing lunch should be distinguished from grabbing food or drinks.
A short kitchen visit with little or no sustained food-preparation
activity provides stronger evidence for grabbing food or drinks.
Longer kitchen occupancy, repeated movement, and food-preparation
behavior provide stronger evidence for preparing lunch.

Preparing lunch should also be distinguished from having fruits or
snacks. A snack is more likely to involve brief food retrieval followed
by relatively stationary presence in an eating or seating location.
Sustained kitchen activity associated with preparing a more substantial
midday meal provides stronger evidence for preparing lunch.

Preparing lunch should be distinguished from washing dishes. Food
preparation, cooking-appliance interaction, and movement associated with
preparing ingredients provide stronger evidence for preparing lunch.
If food preparation has ended and kitchen activity continues after
eating, washing dishes may become more strongly supported.

Preparing lunch should also be distinguished from having lunch. Kitchen
presence with active food preparation provides stronger evidence for
preparing lunch, while movement to a dining or usual eating location
followed by sustained presence and reduced food-preparation activity may
indicate that preparation has ended and lunch consumption has begun.

The temporal transition between these activities is important. For
example, sustained kitchen activity followed by movement to the dining
area may indicate a transition from preparing lunch to having lunch
rather than a single activity continuing throughout the entire window.

Temperature and humidity changes may provide additional contextual
evidence when cooking produces heat or steam. However, environmental
changes may be delayed and should normally be treated as supporting
rather than primary evidence.

If a pet is present, isolated kitchen PIR activation should be
interpreted cautiously. Sustained human movement, recent confirmed human
location, food-preparation appliance interaction, duration of kitchen
occupancy, and the temporal sequence provide stronger evidence that the
resident is preparing lunch.

The resident's historical lunch routine may provide additional context.
For example, recurring midday patterns involving kitchen presence,
food-preparation activity, and subsequent movement to a dining area may
strengthen the classification. Historical patterns should not override
contradictory current sensor observations.

When kitchen activity is detected but the available evidence cannot
reliably determine whether the resident is preparing lunch, preparing
another meal, obtaining a snack, or performing another kitchen activity,
the system should maintain uncertainty and reduce classification
confidence rather than relying only on the time of day.

Evidence for preparing lunch is strongest when midday contextual timing,
sustained or repeated kitchen presence, food-preparation behavior,
optional cooking-appliance activity, and the temporal sequence of sensor
events consistently support preparation of the resident's midday meal.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "preparing_lunch",
    }
)

# having dinner
Document(
    page_content="""
Having dinner refers to the period during which the resident is
consuming their evening meal after or during dinner preparation.

Resident presence in the dining area, kitchen seating area, living room,
or another location normally used for eating provides important
location evidence for having dinner.

A temporal transition from preparing dinner to sustained presence in an
eating location provides strong evidence for having dinner. For example,
sustained kitchen activity involving food preparation followed by
movement to the dining area is consistent with the transition from
preparing dinner to having dinner.

Having dinner generally involves less interaction with cooking and
food-preparation appliances than preparing dinner. Therefore, a
reduction or termination of oven, stove, microwave, air fryer, kettle,
or other cooking-appliance activity after a period of kitchen activity
may support the transition to having dinner.

Some cooking appliances may remain active after the resident begins
eating. For example, an oven or other appliance may continue operating
while the resident is at the dining table. Therefore, continued
appliance activity should not automatically be interpreted as evidence
that the resident is still preparing dinner.

Sustained presence in the dining or eating area with relatively limited
PIR activity may be consistent with having dinner because the resident
may remain seated while eating. Low PIR activity should not be
interpreted as evidence that the resident has left the area.

Occasional PIR activations during dinner may occur when the resident
changes position, stands briefly, returns to the kitchen, retrieves an
item, or moves between the kitchen and dining area.

Time of day provides useful contextual evidence. Presence in a usual
eating location during the resident's typical evening meal period
increases the likelihood of having dinner. However, time of day should
not be used as the sole basis for classification because the resident
may eat dinner earlier or later than usual.

Having dinner should be distinguished from preparing dinner. Sustained
kitchen movement, ingredient preparation, and interaction with cooking
appliances provide stronger evidence for preparing dinner. Movement from
the kitchen to an eating location followed by more stationary behavior
provides stronger evidence for having dinner.

Having dinner should also be distinguished from having fruits or snacks.
Dinner is generally associated with a main evening meal and may be
preceded by a longer period of food preparation. Having fruits or snacks
may involve a shorter retrieval-and-consumption sequence and can occur
at different times throughout the day. These differences should be
treated as contextual rather than strict rules.

Having dinner should be distinguished from resting on the couch or
chairs. Both activities may involve sustained presence and limited PIR
activity. Evidence of preceding dinner preparation, movement to a usual
eating location, evening meal timing, and the resident's meal routine
provides stronger evidence for having dinner.

Having dinner should also be distinguished from watching television.
If the resident normally eats in the living room while the television
is active, both activities may be plausible simultaneously. Television
activity alone should not override strong evidence that the resident is
having dinner.

Current NILM appliance activity is not required for having dinner.
Evidence from the preceding portion of the observation window may be
more informative than the appliance state at the current moment.

For example, recent oven, stove, or microwave activity combined with
sustained kitchen presence, followed by appliance deactivation and
movement to the dining area, provides useful evidence that the resident
has transitioned from preparing dinner to having dinner.

The temporal sequence should therefore be considered across the
observation window rather than evaluating only the most recent sensor
sample.

Temperature and humidity may change during preceding food preparation
but generally provide weak direct evidence for having dinner. Elevated
kitchen temperature or humidity may persist after cooking has ended and
should not be interpreted as evidence that meal preparation is still
occurring.

If a pet is present, isolated PIR activity in the dining, kitchen, or
living area should be interpreted cautiously. Recent confirmed human
location, preceding meal preparation, movement between rooms, duration
of presence, and appliance interactions should be considered together.

The resident's historical dinner routine may provide additional context.
For example, a recurring sequence involving evening kitchen activity,
cooking-appliance use, movement to the dining area, and sustained
presence may strengthen the inference of having dinner. Historical
patterns should not override contradictory current sensor observations.

When the available evidence indicates a sedentary evening activity but
does not reliably distinguish having dinner from resting, watching
television, or another activity, the system should maintain uncertainty
and reduce classification confidence rather than relying solely on time
of day.

Evidence for having dinner is strongest when the temporal sensor
sequence shows preceding dinner preparation, movement from the kitchen
toward the resident's usual eating location, reduced food-preparation
activity, sustained presence with relatively limited movement, and
evening contextual timing consistent with the resident's meal routine.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "having_dinner",
    }
)

# having lunch
Document(
    page_content="""
Having lunch refers to the period during which the resident is consuming
their midday meal after or during lunch preparation.

Resident presence in the dining area, kitchen seating area, living room,
or another location normally used for eating provides important location
evidence for having lunch.

A temporal transition from preparing lunch to sustained presence in an
eating location provides strong evidence for having lunch. For example,
sustained kitchen activity involving food preparation followed by
movement to the dining area is consistent with the transition from
preparing lunch to having lunch.

Having lunch generally involves less interaction with food-preparation
and cooking appliances than preparing lunch. Therefore, a reduction or
termination of microwave, stove, oven, kettle, toaster, or other
food-preparation appliance activity after kitchen activity may support
the transition to having lunch.

Some appliances may remain active after the resident begins eating.
Therefore, continued appliance activity should not automatically be
interpreted as evidence that the resident is still preparing lunch.
Resident location and the temporal sequence should also be considered.

Sustained presence in the dining or eating area with relatively limited
PIR activity may be consistent with having lunch because the resident
may remain seated while eating. Low PIR activity should not be
interpreted as evidence that the resident has left the area.

Occasional PIR activations may occur during lunch when the resident
changes position, stands briefly, returns to the kitchen, retrieves an
item, or moves between the kitchen and dining area.

Time of day provides useful contextual evidence. Presence in a usual
eating location during the resident's typical midday meal period
increases the likelihood of having lunch. However, time of day should
not be used as the sole basis for classification because the resident
may eat lunch earlier or later than usual.

Having lunch should be distinguished from preparing lunch. Sustained
kitchen movement and interaction with food-preparation appliances provide
stronger evidence for preparing lunch. Movement from the kitchen to an
eating location followed by more stationary behavior provides stronger
evidence for having lunch.

Having lunch should also be distinguished from grabbing food or drinks.
Grabbing food or drinks is generally characterized by a relatively short
visit to the kitchen or food-storage area followed by movement away.
Having lunch is more likely to involve a longer period of sustained
presence at an eating location, often following meal preparation.

Having lunch should be distinguished from having fruits or snacks.
Having fruits or snacks may involve a shorter retrieval-and-consumption
sequence and may occur between regular meals. Having lunch is more
strongly supported by midday contextual timing, preceding meal
preparation, a longer eating period, and the resident's regular meal
routine.

These timing and duration differences should be treated as contextual
rather than strict rules because the resident's eating habits may vary.

Having lunch should also be distinguished from resting on the couch or
chairs. Both activities may involve sustained presence and limited PIR
activity. Evidence of preceding food preparation, movement to a usual
eating location, midday meal timing, and the resident's meal routine
provides stronger evidence for having lunch.

If the resident eats lunch while watching television, evidence for both
having lunch and watching television may be present simultaneously.
Television activity should not automatically override strong evidence
of meal consumption. The system should use the activity definitions and
temporal context to determine the most appropriate label.

Current NILM appliance activity is not required for having lunch.
Evidence from the preceding portion of the observation window may be
more informative than appliance activity at the current moment.

For example, recent microwave or other food-preparation appliance
activity combined with kitchen presence, followed by appliance
deactivation and sustained dining-area presence, provides useful evidence
that the resident has transitioned from preparing lunch to having lunch.

Lunch may also involve food that requires no detectable electrical
appliance activity. Therefore, the absence of preceding NILM detections
should not rule out having lunch when location, timing, duration, and
other contextual evidence support it.

The temporal sequence should be considered across the observation window
rather than evaluating only the most recent sensor state.

Temperature and humidity generally provide little direct evidence for
having lunch. Environmental changes caused by preceding food preparation
may persist during the meal and should be treated only as supporting
context.

If a pet is present, isolated PIR activity in the kitchen, dining area,
or living room should be interpreted cautiously. Recent confirmed human
location, preceding food preparation, movement between rooms, duration
of presence, and appliance interactions should be considered together.

The resident's historical lunch routine may provide additional context.
For example, a recurring sequence involving midday kitchen activity,
food preparation, movement to the dining area, and sustained presence
may strengthen the inference of having lunch. Historical patterns should
not override contradictory current sensor observations.

When the available evidence indicates a sedentary midday activity but
does not reliably distinguish having lunch from having a snack, resting,
or another activity, the system should maintain uncertainty and reduce
classification confidence rather than relying solely on time of day.

Evidence for having lunch is strongest when the temporal sensor sequence
shows preceding lunch preparation or food retrieval, movement from the
kitchen toward the resident's usual eating location, reduced
food-preparation activity, sustained presence with relatively limited
movement, and midday contextual timing consistent with the resident's
meal routine.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "having_lunch",
    }
)

# playing board games
Document(
    page_content="""
Playing board games refers to a period during which the resident is
participating in a board game, card game, puzzle-based tabletop game,
or similar non-electronic game.

This activity commonly occurs at a dining table, living-room table,
kitchen table, or another surface suitable for playing a tabletop game.
Resident presence in a location normally used for board games provides
useful contextual evidence.

Playing board games is generally a sedentary activity. The resident may
remain seated for an extended period, so PIR activity may be limited or
intermittent. A lack of frequent PIR activation should not be
interpreted as evidence that the resident has left the area.

Occasional PIR activations may occur when the resident changes position,
reaches for game pieces, stands briefly, moves around the table, or
retrieves an item.

Playing board games generally does not require a distinctive electrical
appliance. Therefore, the absence of NILM-detected appliance activity
should not reduce the likelihood of this activity when location,
duration, and contextual evidence support it.

The absence of television, gaming-console, or computer activity may help
distinguish a non-electronic board game from electronic entertainment,
but the absence of these devices alone is not sufficient evidence that
a board game is being played.

Playing board games should be distinguished from playing video games.
Activation of a gaming console, gaming computer, or other dedicated
gaming equipment provides stronger evidence for playing video games.
Board games are more consistent with sustained presence around a table
or usual tabletop gaming location without gaming-device activity.

Playing board games should also be distinguished from using a PC.
Computer or monitor activity together with presence at the resident's
usual computer location provides stronger evidence for PC use. Board
games generally do not require sustained computer activity.

Playing board games should be distinguished from reading books. Both
activities may involve prolonged seated presence with limited PIR
activity and little or no appliance activity. Location may provide
supporting evidence if the resident has distinct areas normally used
for reading and tabletop games, but location alone may not reliably
separate these activities.

Playing board games should also be distinguished from resting on the
couch or chairs. Resting is more likely to involve prolonged stationary
presence in a usual relaxation location, while board games may involve
presence at a table and intermittent movement associated with interacting
with game materials. Standard PIR sensors may not reliably detect these
small movements, so this distinction may remain uncertain.

Playing board games may occur while the resident is chatting with family
members or visitors. In-person conversation and board-game activity can
therefore occur simultaneously. Evidence that another person is present
may provide useful contextual support for multiplayer board games, but
conversation alone should not be interpreted as evidence that a board
game is being played.

The system should not assume that another person must be present because
some board games, card games, and puzzles can be performed individually.
Likewise, standard PIR sensors generally cannot reliably determine the
number or identity of people in a room.

If household or schedule context explicitly indicates that family
members or visitors are present, sustained activity around a table may
increase the plausibility of playing a board game. Visitor presence
should be based on available evidence or known context rather than
invented from PIR activity.

The temporal sequence may provide supporting evidence. For example, the
resident moving to a dining or living-room table, remaining there for an
extended period with intermittent movement, and later leaving the area
may be consistent with playing a board game.

However, the same sensor pattern may also correspond to eating, reading,
doing paperwork, chatting, or another seated activity. The system should
therefore avoid assigning high confidence to playing board games based
only on PIR patterns.

Meal-related activities should also be considered when board games occur
at a dining table. Recent food preparation followed by dining-area
presence may provide stronger evidence for having a meal, while prolonged
table occupancy after the meal has ended may make another seated activity,
including a board game, more plausible.

Time of day may provide contextual information if the resident has a
regular board-game routine, but time should not be used as the primary
basis for classification.

Temperature and humidity generally provide little direct evidence for
playing board games and should be treated only as environmental context.

If a pet is present, isolated PIR activation near a table or seating area
should be interpreted cautiously. Recent confirmed human location,
duration of presence, movement between rooms, and other available
context should be considered.

The resident's historical activity patterns may provide particularly
useful supporting evidence. For example, if the resident regularly plays
board games with visiting family members at the dining table, a similar
future sensor sequence may increase the plausibility of this activity.
Historical patterns should not override contradictory current sensor
observations.

When available sensor evidence is equally compatible with playing board
games, reading, chatting, resting, or another sedentary activity, the
system should maintain uncertainty and reduce classification confidence
rather than inventing evidence for board-game activity.

Evidence for playing board games is strongest when the resident remains
at a known tabletop gaming location for a sustained period, movement is
consistent with a sedentary but interactive activity, electronic gaming
devices are not providing stronger evidence for another activity, and
household, schedule, or historical context specifically supports
board-game activity.
""",
    metadata={
        "knowledge_type": "general",
        "category": "activity_rule",
        "activity": "playing_board_games",
    }
)

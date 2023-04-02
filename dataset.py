# get carbs:
# select carbs, timestamp 
# from Treatments 
# where carbs <> 0 
# order by timestamp;

# get insulin:
# select insulin, timestamp 
# from Treatments 
# where insulin <> 0 
# order by timestamp;

# get glucose
# select (calculated_value * 0.0555) as glucose, timestamp
# from BgReadings
# order by timestamp;
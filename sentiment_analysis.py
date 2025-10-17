import pymongo
from collections import defaultdict

def get_age_group_mapping():
    """Returns the age group mapping for MongoDB aggregation pipeline"""
    return {
        "$switch": {
            "branches": [
                {"case": {"$in": ["$Age of User", ["0-20", "21-30"]]}, "then": "0-30"},
                {"case": {"$in": ["$Age of User", ["31-45", "46-60"]]}, "then": "31-60"},
                {"case": {"$in": ["$Age of User", ["60-70", "70-100"]]}, "then": "61-100"}
            ],
            "default": "Unknown"
        }
    }

def calculate_sentiment_stats(sentiments_dict):
    """Calculate counts and percentages from a sentiment dictionary"""
    pos_count = sentiments_dict.get("positive", 0)
    neg_count = sentiments_dict.get("negative", 0)
    neu_count = sentiments_dict.get("neutral", 0)
    total = pos_count + neg_count + neu_count

    if total > 0:
        return {
            'pos_count': pos_count,
            'neg_count': neg_count,
            'neu_count': neu_count,
            'total': total,
            'pos_pct': (pos_count / total) * 100,
            'neg_pct': (neg_count / total) * 100,
            'neu_pct': (neu_count / total) * 100
        }
    return None

def print_sentiment_table(data_dict, header_label):
    """Print a formatted sentiment distribution table"""
    print(f"\n{header_label} | Total | Positive % | Negative % | Neutral %")
    print("-" * 65)

    for key in sorted(data_dict.keys()):
        stats = calculate_sentiment_stats(data_dict[key])
        if stats:
            print(f"{key:11s} | {stats['total']:5d} | {stats['pos_pct']:9.1f}% | "
                  f"{stats['neg_pct']:9.1f}% | {stats['neu_pct']:8.1f}%")

def find_peaks(stats_list):
    """Find peak positive, negative, neutral, and most active from stats list"""
    if not stats_list:
        return None

    return {
        'most_positive': max(stats_list, key=lambda x: x['pos_pct']),
        'most_negative': max(stats_list, key=lambda x: x['neg_pct']),
        'most_neutral': max(stats_list, key=lambda x: x['neu_pct']),
        'most_active': max(stats_list, key=lambda x: x['total'])
    }

def process_aggregation_results(results):
    """Process MongoDB aggregation results into a nested dictionary"""
    data_dict = defaultdict(lambda: defaultdict(int))
    for result in results:
        keys = result["_id"]
        count = result["count"]
        # Handle both 2-level (time/age + sentiment) and 3-level (age + time + sentiment) groupings
        if len(keys) == 2:
            primary_key = list(keys.values())[0]
            sentiment = keys["sentiment"]
            data_dict[primary_key][sentiment] = count
        elif len(keys) == 3:
            age_group = keys["age_group"]
            time_period = keys["time_period"]
            sentiment = keys["sentiment"]
            if not isinstance(data_dict[age_group], defaultdict):
                data_dict[age_group] = defaultdict(lambda: defaultdict(int))
            data_dict[age_group][time_period][sentiment] = count
    return data_dict

# ===================================================================
# MONGODB CONNECTION
# ===================================================================

uri = "mongodb+srv://saad:mongoPass@dataprojectid2221.2uplah7.mongodb.net/?retryWrites=true&w=majority&appName=DataProjectID2221"
client = pymongo.MongoClient(uri)
db = client["Our_Database"]
processed_collection = db["processed_sentiment_data"]

print("="*60)
print("=== SENTIMENT ANALYSIS ===")
print("="*60)
print(f"Analyzing {processed_collection.count_documents({})} documents from processed collection")

# ===================================================================
# SENTIMENT ANALYSIS BY TIME PERIOD
# ===================================================================
print("\n" + "="*50)
print("=== SENTIMENT ANALYSIS BY TIME PERIOD ===")
print("="*50)

time_period_pipeline = [
    {
        "$group": {
            "_id": {
                "time_period": "$Time of Tweet",
                "sentiment": "$sentiment"
            },
            "count": {"$sum": 1}
        }
    },
    {
        "$sort": {"_id.time_period": 1, "_id.sentiment": 1}
    }
]

time_results = list(processed_collection.aggregate(time_period_pipeline))
time_period_data = defaultdict(lambda: defaultdict(int))

for result in time_results:
    time_period = result["_id"]["time_period"]
    sentiment = result["_id"]["sentiment"]
    count = result["count"]
    time_period_data[time_period][sentiment] = count

print_sentiment_table(time_period_data, "Time Period")

print("\n=== Peak Sentiment Times Analysis ===")
time_stats = []
for time_period in time_period_data.keys():
    stats = calculate_sentiment_stats(time_period_data[time_period])
    if stats:
        stats['period'] = time_period
        time_stats.append(stats)

peaks = find_peaks(time_stats)
if peaks:
    print(f"\nPeak Positive Sentiment Time: {peaks['most_positive']['period'].upper()}")
    print(f"   - {peaks['most_positive']['pos_pct']:.1f}% positive ({peaks['most_positive']['pos_count']} out of {peaks['most_positive']['total']} tweets)")

    print(f"\nPeak Negative Sentiment Time: {peaks['most_negative']['period'].upper()}")
    print(f"   - {peaks['most_negative']['neg_pct']:.1f}% negative ({peaks['most_negative']['neg_count']} out of {peaks['most_negative']['total']} tweets)")

    print(f"\nMost Active Time Period: {peaks['most_active']['period'].upper()}")
    print(f"   - {peaks['most_active']['total']} total tweets")

print("\n=== Time Analysis Complete ===")

# ===================================================================
# SENTIMENT ANALYSIS BY AGE DEMOGRAPHICS
# ===================================================================
print("\n" + "="*60)
print("=== SENTIMENT ANALYSIS BY AGE DEMOGRAPHICS ===")
print("="*60)

age_pipeline = [
    {
        "$addFields": {
            "age_group": get_age_group_mapping()
        }
    },
    {
        "$group": {
            "_id": {
                "age": "$age_group",
                "sentiment": "$sentiment"
            },
            "count": {"$sum": 1}
        }
    },
    {
        "$sort": {"_id.age": 1, "_id.sentiment": 1}
    }
]

age_results = list(processed_collection.aggregate(age_pipeline))
age_data = defaultdict(lambda: defaultdict(int))

for result in age_results:
    age = result["_id"]["age"]
    sentiment = result["_id"]["sentiment"]
    count = result["count"]
    age_data[age][sentiment] = count

print_sentiment_table(age_data, "Age")

print("\n=== Comparing Sentiment Patterns Across Age Groups ===")
age_stats = []
for age in age_data.keys():
    stats = calculate_sentiment_stats(age_data[age])
    if stats:
        stats['age_group'] = age
        age_stats.append(stats)

age_peaks = find_peaks(age_stats)
if age_peaks:
    print(f"\nMost Positive Age Group: {age_peaks['most_positive']['age_group']}")
    print(f"   - {age_peaks['most_positive']['pos_pct']:.1f}% positive sentiment")

    print(f"\nMost Negative Age Group: {age_peaks['most_negative']['age_group']}")
    print(f"   - {age_peaks['most_negative']['neg_pct']:.1f}% negative sentiment")

    print(f"\nMost Active Age Group: {age_peaks['most_active']['age_group']}")
    print(f"   - {age_peaks['most_active']['total']} tweets")

    youngest = [a for a in age_stats if '0-30' in a['age_group']]
    oldest = [a for a in age_stats if '61-100' in a['age_group']]

    if youngest and oldest:
        pos_diff = youngest[0]['pos_pct'] - oldest[0]['pos_pct']
        neg_diff = youngest[0]['neg_pct'] - oldest[0]['neg_pct']

        print(f"\nComparison: Youngest (0-30) vs Oldest (61-100)")
        print(f"   - Positive sentiment difference: {pos_diff:+.1f}%")
        print(f"   - Negative sentiment difference: {neg_diff:+.1f}%")
        print(f"   -> {'Younger' if pos_diff > 0 else 'Older'} users are more positive")

print("\n=== Age Demographics Analysis Complete ===")

# ===================================================================
# CORRELATION BETWEEN AGE AND SENTIMENT TIMING
# ===================================================================
print("\n" + "="*60)
print("=== AGE AND TIME CORRELATION ANALYSIS ===")
print("="*60)

age_time_pipeline = [
    {
        "$addFields": {
            "age_group": get_age_group_mapping()
        }
    },
    {
        "$group": {
            "_id": {
                "age_group": "$age_group",
                "time_period": "$Time of Tweet",
                "sentiment": "$sentiment"
            },
            "count": {"$sum": 1}
        }
    },
    {
        "$sort": {"_id.age_group": 1, "_id.time_period": 1, "_id.sentiment": 1}
    }
]

age_time_results = list(processed_collection.aggregate(age_time_pipeline))
age_time_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for result in age_time_results:
    age_group = result["_id"]["age_group"]
    time_period = result["_id"]["time_period"]
    sentiment = result["_id"]["sentiment"]
    count = result["count"]
    age_time_data[age_group][time_period][sentiment] = count

print("\n=== Sentiment by Age Group and Time Period ===")
for age_group in sorted(age_time_data.keys()):
    print(f"\n{age_group} Age Group:")
    print("  Time Period | Total | Positive % | Negative % | Neutral %")
    print("  " + "-" * 60)

    for time_period in sorted(age_time_data[age_group].keys()):
        stats = calculate_sentiment_stats(age_time_data[age_group][time_period])
        if stats:
            print(f"  {time_period:11s} | {stats['total']:5d} | {stats['pos_pct']:9.1f}% | {stats['neg_pct']:9.1f}% | {stats['neu_pct']:8.1f}%")

print("\n=== Sentiment Variation by Time for Each Age Group ===")

for age_group in sorted(age_time_data.keys()):
    time_sentiments = []

    for time_period in age_time_data[age_group].keys():
        stats = calculate_sentiment_stats(age_time_data[age_group][time_period])
        if stats:
            stats['time'] = time_period
            time_sentiments.append(stats)

    if len(time_sentiments) > 1:
        most_pos_time = max(time_sentiments, key=lambda x: x['pos_pct'])
        least_pos_time = min(time_sentiments, key=lambda x: x['pos_pct'])

        print(f"\n{age_group} Age Group:")
        print(f"  Most positive at: {most_pos_time['time']} ({most_pos_time['pos_pct']:.1f}%)")
        print(f"  Least positive at: {least_pos_time['time']} ({least_pos_time['pos_pct']:.1f}%)")
        print(f"  Sentiment variation: {most_pos_time['pos_pct'] - least_pos_time['pos_pct']:.1f}% difference")

print("\n=== Age-Time Correlation Analysis Complete ===")

client.close()
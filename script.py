from os.path import abspath
import numpy as np
import pandas as pd
import time
import datetime
import pytz
import re
import math
import sys
import getopt

ORIGINAL_DATA_PATH = sys.argv[1]
PROCESSED_DATA_PATH = sys.argv[2]


def make_timestamp(day, month, year, min=None, hour=None, format='%d/%m/%Y %H:%M:%S', tz='UTC'):
    tz = pytz.timezone(tz)
    return datetime.datetime(year, month, day, hour, min, 0, 0, tz)


def handle_tags(x):
    tags = []
    uploaded_via = []
    foursquare_venue = []

    uploaded_str = 'uploaded:by='
    fourquare_str = 'foursquare:venue='

    tag = str(x).split(',')
    for i, t in enumerate(tag):
        if uploaded_str in t:
            uploaded_via.append(t.replace(uploaded_str, ''))
        elif fourquare_str in t:
            foursquare_venue.append(t.replace(fourquare_str, ''))
        else:
            tags.append(t)

    return ','.join(tags), ''.join(uploaded_via), ''.join(foursquare_venue)


def handle_title(x):
    if type(x) == float:
        return np.nan, np.nan, np.nan, np.nan
    tags = re.findall(r'#(\w+)', x)
    links = re.findall(
        r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', x)
    file_title = re.findall(r'^(?!@|#)(?=.*_)([^\s]+)', x)
    people_tag = re.findall(r'@(\w+)', x)

    return ','.join(tags), ','.join(links), ','.join(file_title), ','.join(people_tag)


###############

data_file_path = abspath(ORIGINAL_DATA_PATH)
data_header_dtypes = {'id'}

data = pd.read_csv(data_file_path, error_bad_lines=False,
                   encoding='utf8', sep=',')

print('Original data loaded via : ', ORIGINAL_DATA_PATH)
print('Start processing...')

###############


# Remove blanks chars in labels :
data.columns = [x.strip() for x in data.columns]

# Creates dates as timestamps :
data['date_upload'] = data.apply(lambda x: make_timestamp(x['date_upload_day'], x['date_upload_month'],
                                                          x['date_upload_year'], x['date_upload_minute'], x['date_upload_hour']), axis=1)
data['date_taken'] = data.apply(lambda x: make_timestamp(x['date_taken_day'], x['date_taken_month'],
                                                         x['date_taken_year'], x['date_upload_minute'], x['date_taken_hour']), axis=1)

# Drop duplicated values :
data.drop_duplicates(subset=['id', 'user'], inplace=True)

date_related_columns = ['date_taken_minute', 'date_taken_hour', 'date_taken_day', 'date_taken_month', 'date_taken_year',
                        'date_upload_minute', 'date_upload_hour', 'date_upload_day', 'date_upload_month', 'date_upload_year']

# Drop abnormal dt values :
dt_abnormal_idx = data[data['date_upload'] < data['date_taken']].index
data.drop(dt_abnormal_idx, inplace=True)

# Add bins to datetime values :
bins = [0, 8, 11, 13, 17, 22, 24]
time_labels = ['early morning', 'morning',
               'midday', 'afternoon', 'evening', 'night']
data['date_taken_bin'] = pd.cut(
    x=data['date_taken'].dt.hour, bins=bins, labels=time_labels)
data['date_upload_bin'] = pd.cut(
    x=data['date_upload'].dt.hour, bins=bins, labels=time_labels)

data.drop(date_related_columns, inplace=True, axis=1)

# Apply handle_tags fn :
data[['tags', 'uploaded_via', 'foursquare_venue']] = data.apply(
    lambda x: handle_tags(x['tags']), axis=1, result_type='expand')

# Apply handle_title fn :
data[['title_tags', 'links', 'file_title', 'people_tag']] = data.apply(
    lambda x: handle_title(x['title']), axis=1, result_type='expand')
data['tags'] = data['tags'] + ',' + data['title_tags']
data.drop(['title_tags'], axis=1, inplace=True)

# Replace textual nan by np.nan
data['tags'].replace('nan,', np.nan, inplace=True)

###############

data.to_csv(PROCESSED_DATA_PATH, sep=',', index=False)
print('Processed data stored : ', PROCESSED_DATA_PATH)


###############


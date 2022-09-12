import os
import time
import numpy as np
import pandas as pd
from glob import glob
import holidays


def load_POI_location():
    zip_csv = []
    for i in range(5):
        zip_csv.append(pd.read_csv(f'./CORE_POI-2019_03-2020_03_25/core_poi-part{i + 1}.csv.gz', compression='gzip'))
    US_POI_2019 = pd.concat(zip_csv, axis=0)
    print('US POI 2019 shape:', US_POI_2019.shape)
    return US_POI_2019


def NAICS2category(POI_data, cat_list):
    def naics2cat(naics_code):
        try:
            naics_str = str(int(naics_code))
        except:
            return 'NaN'
        if naics_str.startswith('445'):
            cat = 'grocery_store'
        elif (naics_str.startswith('441')) | (naics_str.startswith('442')) | (naics_str.startswith('443')) | (
        naics_str.startswith('444')) | (naics_str.startswith('446')) | (naics_str.startswith('447')) | (
        naics_str.startswith('448')) | (naics_str.startswith('45')):
            cat = 'other_retailer'
        elif naics_str.startswith('48'):
            cat = 'transportation'
        elif (naics_str.startswith('51')) | (naics_str.startswith('52')) | (naics_str.startswith('53')) | (
        naics_str.startswith('54')) | (naics_str.startswith('55')) | (naics_str.startswith('561')) | (
        naics_str.startswith('92')):
            cat = 'office'
        elif naics_str.startswith('61'):
            cat = 'school'
        elif naics_str.startswith('62'):
            cat = 'healthcare'
        elif naics_str.startswith('71'):
            cat = 'entertainment'
        elif naics_str.startswith('721'):
            cat = 'hotel'
        elif naics_str.startswith('722'):
            cat = 'restaurant'
        elif naics_str.startswith('81'):
            cat = 'service'
        else:
            cat = 'other'
        return cat

    POI_data['category'] = POI_data['naics_code'].apply(naics2cat)
    for cat in cat_list:
        print(cat, POI_data[POI_data['category']==cat].shape[0])

    id2cat = POI_data[['safegraph_place_id', 'category']]
    loc_name2cat = POI_data[['location_name', 'postal_code', 'category']]    # loc_name2cat exists one2many relations
    return id2cat, loc_name2cat


def process_SG_POI(id2cat:pd.DataFrame, loc_name2cat:pd.DataFrame):     # zip files to .npz

    for file_dir in sorted(glob(os.path.join(poi_dir, '*.gz'))):
        zip_file = file_dir.split('\\')[-1]
        print(time.ctime())
        print('    Processing: ', zip_file)
        zip_date = zip_file[:10]

        # initialize a zero tensor
        week_tensor = np.zeros((24 * 7, len(us_states), len(naics_categories)))
        total_record = 0
        valid_record = 0
        unknown_loc = 0
        unknown_cat = 0

        # load zipped csv file
        zip_csv = pd.read_csv(file_dir, compression='gzip')
        zip_csv_ = pd.merge(zip_csv, id2cat, on='safegraph_place_id', how='outer', indicator=True)
        zip_csv_by_id = zip_csv_[zip_csv_['_merge'] == 'both']    # joined by place_id
        zip_csv_left = zip_csv_[zip_csv_['_merge'] == 'left_only']
        zip_csv_by_loc_name = pd.merge(zip_csv_left, loc_name2cat, on=['location_name', 'postal_code'], how='inner')      # join the left with [loc_name, zipcode]
        zip_csv_valid = pd.concat([zip_csv_by_id, zip_csv_by_loc_name], ignore_index=True)
        print('Finished merging!')
        print('#Before:', zip_csv.shape[0], '#After merging place_id:', zip_csv_by_id.shape[0], '#After merging loc_name', zip_csv_valid.shape[0])
        total_record += zip_csv.shape[0]
        valid_record += zip_csv_valid.shape[0]

        for i in range(zip_csv_valid.shape[0]):  # loop through each row
            item = zip_csv_valid.iloc[i]

            # location
            try:
                loc_id = us_states.index(item['region'])
            except:
                # print(item['region'])
                unknown_loc += 1
                continue

            # category
            try:
                cat_id = naics_categories.index(item['category'])
            except:
                # print(item['location_name'])
                unknown_cat += 1
                continue

            # TS
            TS = []
            for i, value in enumerate(item['visits_by_each_hour'].split(',')):
                if i == 0:
                    TS.append(int(value[1:]))
                elif i == 167:
                    TS.append(int(value[:-1]))
                else:
                    TS.append(int(value))

            for t in range(24 * 7):
                week_tensor[t, loc_id, cat_id] += TS[t]

        # save each file
        stats = np.array([total_record, valid_record, unknown_loc, unknown_cat])
        np.savez_compressed(f'{poi_dir}POI-visit-week-{zip_date}.npz',
                            stats=stats,
                            visit=week_tensor)
    return


def get_ST_tensor(poi_dir):
    print('Processing dir', poi_dir)

    st_tensor = []
    for file_dir in sorted(glob(os.path.join(poi_dir, '*.npz'))):
        print('Loading:', file_dir.split('\\')[-1])
        npz_data = np.load(file_dir)
        print('#Valid records:', npz_data['stats'][2])
        US_POI_week_visit = npz_data['visit'][:, :, :10]

        st_tensor.append(US_POI_week_visit)

    st_tensor = np.concatenate(st_tensor, axis=0)
    print(st_tensor.shape)

    return st_tensor


# onehot encode temporal covariates
def get_metadata(start:str, end:str, freq:str='3H'):
    date_information = pd.DataFrame({'datetime': pd.date_range(start=start, end=end, freq=freq)})
    date_information.drop([date_information.shape[0]-1], inplace=True)
    date_information['hour'] = date_information.datetime.dt.time        # time of day
    date_information['day'] = date_information.datetime.dt.dayofweek
    date_information['month'] = date_information.datetime.dt.month

    f = lambda x: (x.date() in holidays.US()) * 1
    date_information['holiday'] = date_information.datetime.apply(f)
    date_information.loc[(date_information.day == 5) | (date_information.day == 6), 'holiday'] = 1
    date_information.set_index('datetime', inplace=True)

    #date_information.to_csv('../../data/day_information.csv', index=False)         # numerical
    date_information = date_information.astype('str')
    date_information = pd.get_dummies(date_information)
    date_information.drop(['holiday_0'], axis=1, inplace=True)
    #date_information.to_csv(out_dir+'/day_information_onehot.csv', index=False)     # one_hot
    date_information = date_information.to_numpy()
    print(date_information.shape)

    return date_information


def get_mask(st_tensor:np.array):
    assert len(st_tensor.shape) == 4

    mask = []
    counter = 0
    for h in range(st_tensor.shape[1]):
        for w in range(st_tensor.shape[2]):
            num_non_zeros = np.count_nonzero(st_tensor[:,h,w,:])
            if num_non_zeros < 1000:
                print(f'({h}, {w}): {num_non_zeros}')
                mask.append((h, w))
                counter += 1
    print('#Masked grids:', counter)

    return mask



poi_dir = './2018_12_31-2020_06_14/'    # put downloaded SafeGraph Weekly Patterns data (.csv.gz) in this dir
naics_categories = ['grocery_store', 'other_retailer', 'transportation', 'office', 'school', 'healthcare',
                    'entertainment', 'hotel', 'restaurant', 'service', 'other', 'NaN']
us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
             'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR',
             'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

if __name__ == '__main__':
    # preprocess to weekly POI visits (.npz)
    POI_data = load_POI_location()
    id2cat, loc_name2cat = NAICS2category(POI_data, naics_categories)
    process_SG_POI(id2cat, loc_name2cat)

    # get ST-tensor
    st_tensor = get_ST_tensor(poi_dir)
    # filter on selected dates
    date_range = pd.date_range(start='20181231', end='20200608', freq='1H').strftime('%Y%m%d').tolist()
    print(len(date_range))
    start, end = date_range.index('20191114'), date_range.index('20200601')
    poi_tensor = st_tensor[start:end, ...]
    print('US POI data shape:', poi_tensor.shape)

    # get onehot-coded temporal coraviates
    tcov = get_metadata(start='20191114', end='20200601', freq='1H')

    # mask not needed
    mask = []

    # save
    np.savez_compressed('../data/COVID-US-51x1-20191114-20200531.npz',
                        poi=poi_tensor,
                        meta_onehot=tcov,
                        mask=mask)
    
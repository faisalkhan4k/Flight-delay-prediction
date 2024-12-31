import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from views import preprocess


def create_model():

    msg=""
    acc=0
    msg_pp=preprocess()

    if(msg_pp=="valid"):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # Importing the necessary files
        df = pd.read_csv('flight_data.csv')
        planes = pd.read_csv('planes.csv')
        airports = pd.read_csv('airports.csv')
        carriers = pd.read_csv('carriers.csv')

        # Dropping the rows that have NaN i.e. NULL values in them
        df = df.dropna()

        # Type casting
        df['dep_time'] = df['dep_time'].astype('int64')
        df['dep_delay'] = df['dep_delay'].astype('int64')
        df['arr_time'] = df['arr_time'].astype('int64')
        df['arr_delay'] = df['arr_delay'].astype('int64')

        def get_stats(group):
            return {'min': group.min(), 'max': group.max(),
                    'count': group.count(), 'mean': group.mean()}

        # _______________________________________________________________
        # Creation of a dataframe with statitical infos on each airline:
        global_stats = df['dep_delay'].groupby(df['carrier']).apply(get_stats).unstack()
        global_stats = global_stats.sort_values('count')

        # ___________________________________________
        # graphs on flights, airports & delays
        global_stats1 = global_stats
        global_stats = global_stats1.head(14)
        codes = global_stats.index.tolist()
        carriers1 = carriers[carriers['IATA_CODE'].isin(codes)]
        abbr_companies = carriers1.set_index('IATA_CODE')['AIRLINE'].to_dict()

        df['origin'].value_counts().to_frame()

        def map_labels(delays):
            if delays > 15:
                return 1
            else:
                return 0

        df['delayed'] = ((df['dep_delay'].map(map_labels) + df['arr_delay'].map(map_labels)) != 0).astype(int)
        df['delayed'].value_counts(normalize=True)

        # feature omission
        columns_to_remove = ['dep_time', 'sched_dep_time', 'dep_delay', 'arr_time', 'sched_arr_time', 'arr_delay',
                             'flight', 'tailnum', 'air_time', 'distance', 'hour', 'minute', 'time_hour']
        df.drop(columns_to_remove, axis=1, inplace=True)

        df['delayed'].value_counts().to_frame()

        df['dest'].value_counts().to_frame()

        df.to_csv('Processed_data15.csv', index=False)

        # Import dataset
        df = pd.read_csv('Processed_data15.csv')

        # Label Encoding
        le_carrier = LabelEncoder()
        df['carrier'] = le_carrier.fit_transform(df['carrier'])

        le_dest = LabelEncoder()
        df['dest'] = le_dest.fit_transform(df['dest'])

        le_origin = LabelEncoder()
        df['origin'] = le_origin.fit_transform(df['origin'])
        df.to_csv('processed_data.csv', index=False)
        # Converting Pandas DataFrame into a Numpy array
        X = df.iloc[:, 0:6].values  # from column(years) to column(distance)
        y = df['delayed'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=18)  # 70% training and 30% test
        # For 75% Train and 25% test use Random state =809

        # Create a Random Forest Classifier
        clf = RandomForestClassifier(random_state=18)
        clf.fit(X_train, y_train)
        ypred=clf.predict(X_test)
        accuracy_RF = accuracy_score(y_test, ypred) * 100.0
        print("Accuracy of Random Forest", accuracy_RF)

        joblib.dump(clf, 'Flight_RF.pkl')
        msg = "Model created successfully"
        print(accuracy_RF)
        return msg, accuracy_RF
    else:
        msg="Model could not be created as dataset is corrupted. Please reload the dataset"
    return msg, acc

#create_model()




import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


#Clean data 

train = pd.read_csv("/Users/stuartbladon/Documents/Duke 2024/AIPI 520/Kaggle/for_students/train.csv")

account = pd.read_csv("/Users/stuartbladon/Documents/Duke 2024/AIPI 520/Kaggle/for_students/account.csv", encoding='ISO-8859-1')
subs = pd.read_csv("/Users/stuartbladon/Documents/Duke 2024/AIPI 520/Kaggle/for_students/subscriptions.csv")
zips = pd.read_csv("/Users/stuartbladon/Documents/Duke 2024/AIPI 520/Kaggle/for_students/zipcodes.csv")
tickets = pd.read_csv("/Users/stuartbladon/Documents/Duke 2024/AIPI 520/Kaggle/for_students/tickets_all.csv")

def clean_to_zero(df, col):
    """takes a pandas df and column name and fills na with zeroes"""
    df[col] = df[col].fillna(0)

def clean_to_mean(df,col):
    """takes a df and fills na with the mean value"""
    df[col] = df[col].fillna(df[col].mean())

for col in account.columns:
    clean_to_zero(account, col)

for col in subs.columns:
    clean_to_zero(subs, col)

for col in tickets.columns:
    clean_to_zero(tickets, col)


# This is more art than science

clean_to_mean(zips, "TaxReturnsFiled")
clean_to_mean(zips, "EstimatedPopulation")
clean_to_mean(zips, "TotalWages")


def since_first_donation(df):
    """changes variable to tie since first donation in days"""
    df['first.donated'] = pd.to_datetime(df['first.donated'])

    assumptive_date = pd.Timestamp('2014-01-01')

    df['first.donated'] = (assumptive_date - df['first.donated']).dt.days

    return df

def loc_info_feat(df, zip):
    """turns zipcode into useful information"""

    zip['percap'] = np.where(
    zip['TotalWages'] == 0, 
    0,  
    zip['TotalWages'] / zip['TaxReturnsFiled']
)
    
    df['Zipcode'] = df['billing.zip.code'].astype('string')
    zip['Zipcode'] = zip['Zipcode'].astype('string')
    
    df = pd.merge(df, zip[['percap', 'Zipcode','Lat','Long']], on='Zipcode', how='left')
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce').fillna(df['Lat'].mean())
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce').fillna(df['Long'].mean())
    
    df.drop(['Zipcode', 'billing.zip.code'], inplace=True, axis=1)
    
    return df

def no_tickets(df, ticket_df):
    """number of tickets purchased (lifetime)"""
    id_counts = ticket_df['account.id'].value_counts()
    df['no_tickets'] = df['account.id'].map(id_counts).fillna(0).astype(int)

    return df

def no_season_tickets(df, ticket_df):
    """number of season tickets purchased (lifetime)"""
    id_counts = ticket_df['account.id'].value_counts()
    df['no_season_tickets'] = df['account.id'].map(id_counts).fillna(0).astype(int)

    return df
def total_info(df):
    """count of non zero variables, represents how much info we have on this person"""
    df['Total_info'] = (df != 0).sum(axis=1)
    df['Total_info'] = df['Total_info'].fillna(0)

    return df
def one_hot_city(df):
    """one hot encoder for billing city"""
 
    encoder = OneHotEncoder(sparse_output=False)

    df["billing.city"] = df["billing.city"].astype('string')
    one_hot_array = encoder.fit_transform(df[['billing.city']])

    one_hot = pd.DataFrame(one_hot_array, columns=encoder.get_feature_names_out(['billing.city']))
    one_hot.index = df.index
    
    df = pd.concat([df, one_hot], axis=1)
    
    return df
def avg_donation(df):
    """average donation through lifetime"""
    df['avg.donation'] = df['amount.donated.lifetime'] / df['no.donations.lifetime']
    df['avg.donation'] = df['avg.donation'].fillna(0)

    return df
def avg_price_level(df, ticketsdf):
    """average price level of tickets purchased"""
    ticketsdf['price.level'] = pd.to_numeric(ticketsdf['price.level'], errors='coerce')
    average_by_name = ticketsdf.groupby('account.id')['price.level'].mean()
    df = pd.merge(df, average_by_name, on = 'account.id', how = 'left')
    df['price.level'] = df['price.level'].fillna(0)


    return df
def avg_price_level_sub(df, ticketsdf):
    """average price level of subscriptions purchased"""
    ticketsdf['price.level'] = pd.to_numeric(ticketsdf['price.level'], errors='coerce')
    average_by_name = ticketsdf.groupby('account.id')['price.level'].mean().reset_index()
    average_by_name.rename(columns={'price.level': 'sub.price.level'}, inplace=True)
    df = pd.merge(df, average_by_name, on = 'account.id', how = 'left')
    df['sub.price.level'] = df['sub.price.level'].fillna(0)


    return df

def avg_sub_tier(df, ticketsdf):
    """average subscription tier"""
    ticketsdf['price.level'] = pd.to_numeric(ticketsdf['subscription_tier'], errors='coerce')
    average_by_name = ticketsdf.groupby('account.id')['subscription_tier'].mean().reset_index()
    df = pd.merge(df, average_by_name, on = 'account.id', how = 'left')
    df['subscription_tier'] = df['subscription_tier'].fillna(0)


    return df

def avg_no_seats(df, ticketsdf):
    """average number of seats bought"""
    ticketsdf['no.seats'] = pd.to_numeric(ticketsdf['no.seats'], errors='coerce')
    average_by_name = ticketsdf.groupby('account.id')['no.seats'].mean().reset_index()
    df = pd.merge(df, average_by_name, on = 'account.id', how = 'left')
    df['no.seats'] = df['no.seats'].fillna(0)


    return df

def avg_no_seats_sub(df, ticketsdf):
    """average number of seats subscribed"""
    ticketsdf['no.seats'] = pd.to_numeric(ticketsdf['no.seats'], errors='coerce')
    average_by_name = ticketsdf.groupby('account.id')['no.seats'].mean().reset_index()
    average_by_name.rename(columns={'no.seats': 'no.seats.sub'}, inplace=True)
    df = pd.merge(df, average_by_name, on = 'account.id', how = 'left')
    df['no.seats.sub'] = df['no.seats.sub'].fillna(0)


    return df

def current_sub_holder(df, subsdf):
    """binary variables for subscription over last 3 years"""

    filtered_ids_13 = subsdf[subsdf['season'] == "2013-2014"]['account.id']
    filtered_ids_12 = subsdf[subsdf['season'] == "2012-2013"]['account.id']
    filtered_ids_11 = subsdf[subsdf['season'] == "2011-2012"]['account.id']
    df['current_season_ticket'] = df['account.id'].isin(filtered_ids_13).astype(int)
    df['12-13'] = df['account.id'].isin(filtered_ids_12).astype(int)
    df['11-12'] = df['account.id'].isin(filtered_ids_11).astype(int)

    return df

def tickets_per_year(df, ticketsdf):
    """tickets bought per year last 3 years"""

    filtered_df_13 = ticketsdf[ticketsdf['season'] == "2013-2014"]["account.id"]
    filtered_df_12 = ticketsdf[ticketsdf['season'] == "2012-2013"]["account.id"]
    filtered_df_11 = ticketsdf[ticketsdf['season'] == "2011-2012"]["account.id"]
    df['tickets_bought_last_year'] = df["account.id"].isin(filtered_df_13).astype(int)
    df['tickets_bought_12'] = df["account.id"].isin(filtered_df_12).astype(int)
    df['tickets_bought_11'] = df["account.id"].isin(filtered_df_11).astype(int)

    return df

def multiple_subs(df, subsdf):
    """do they purchase multiple subscriptions?"""

    filtered = subsdf[subsdf['multiple.subs'] == 'yes']
    df['multiple.subs'] = df['account.id'].isin(filtered).astype(int)

    return df

def feature_engineering(account):
    """implement above features"""

    features = account[["account.id","amount.donated.2013",
                        "amount.donated.lifetime",
                        "no.donations.lifetime",
                        "first.donated",
                        "billing.zip.code","billing.city"]]

    features = since_first_donation(features)
    features = loc_info_feat(features, zips)
    features = no_tickets(features, tickets)
    features = no_season_tickets(features, subs)
    features = total_info(features)
    features = one_hot_city(features)
    features = avg_donation(features)
    features = features.drop(labels=['billing.city'], axis = 1)
    features = avg_price_level(features, tickets)
    features = avg_price_level_sub(features, subs)
    features = avg_sub_tier(features, subs)
    features = avg_no_seats(features, tickets)
    features = avg_no_seats_sub(features, subs)
    features = current_sub_holder(features, subs)
    features = tickets_per_year(features, tickets)
    features = multiple_subs(features, subs)

    data = pd.merge(train, features, on = 'account.id', how = 'left')

    y = data["label"]
    X = data.drop(labels = ["label", "account.id"], axis = 1, inplace = False)
    clean_to_mean(X,"percap")

    return X, y, features
    
#models that survived after trying almost every model under the sun in ensemble, hyperparamaters were decided through cross validation

model2 = RandomForestClassifier(n_estimators = 60)

model5 = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=10, random_state=0)

model8 = XGBClassifier(n_estimators=60, learning_rate=0.1, max_depth=8, random_state=0)

final_model = LogisticRegression(penalty='l2') # meta model to ensemble

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


def training_loop(X, y):
    """train models on data and store"""
    model2.fit(X,y)
    model5.fit(X,y)
    model8.fit(X,y)

    y_hat2 = model2.predict_proba(X)[:, 1]

    y_hat5 = model5.predict_proba(X)[:, 1]

    y_hat8 = model8.predict_proba(X)[:,1]

    y = y.reset_index(drop=True)
    y_hat2 = pd.Series(y_hat2).reset_index(drop=True)
    y_hat5 = pd.Series(y_hat5).reset_index(drop=True)
    y_hat8 = pd.Series(y_hat8).reset_index(drop=True)

    predictions_df = pd.DataFrame({
        'RandomForest_Prediction': y_hat2,
        'GBoost_prediction': y_hat5,
        'XGBoost_prediction': y_hat8,
        'Actual': y
    })

    X = predictions_df[[
                        "RandomForest_Prediction",
                        "GBoost_prediction",
                        "XGBoost_prediction",
                        ]]
    y = predictions_df['Actual']

    X_interactions = poly.fit_transform(X)

    final_model.fit(X_interactions,y)

    print(final_model.coef_, "Coefs")



def test_loop(X_test, y_test):

    """test models"""

    y_hat_2 = model2.predict_proba(X_test)[:, 1]

    y_hat_5 = model5.predict_proba(X_test)[:,1]

    y_hat_8 = model8.predict_proba(X_test)[:,1]


    predictions_test = pd.DataFrame({
        'RandomForest_Prediction': y_hat_2,
        'GBoost_prediction': y_hat_5,
        'XGBoost_prediction': y_hat_8,
    })

    X_t = predictions_test[[
                            "RandomForest_Prediction",
                            'GBoost_prediction',
                            "XGBoost_prediction",
                            ]]
    X_interactions = poly.fit_transform(X_t)

    y_hat_fin = final_model.predict_proba(X_interactions)[:,1]


    meta_acc = roc_auc_score(y_test, y_hat_fin)
    RF_acc = roc_auc_score(y_test, y_hat_2)
    #NN_acc = roc_auc_score(y_test, y_hat_4)
    GB_acc = roc_auc_score(y_test, y_hat_5)
    #NB_acc = roc_auc_score(y_test, y_hat_6)
    XGB_acc = roc_auc_score(y_test, y_hat_8)

    print(meta_acc, RF_acc, GB_acc, XGB_acc)

def create_output(features):
    
    """create output in desired format"""

    test = pd.read_csv("/Users/stuartbladon/Documents/Duke 2024/AIPI 520/Kaggle/for_students/test.csv")
    test['account.id'] = test['ID']

    data_test = pd.merge(test, features, on = 'account.id', how = 'inner')
    print(len(data_test))
    clean_to_mean(data_test,"percap")

    X = data_test.drop(labels = ["account.id","ID"], axis = 1, inplace = False)

    y_hat_two = model2.predict_proba(X)[:, 1]

    y_hat_five = model5.predict_proba(X)[:,1]

    y_hat_eight = model8.predict_proba(X)[:,1]

    predictions_test = pd.DataFrame({
        'RandomForest_Prediction': y_hat_two,
        'GBoost_prediction': y_hat_five,
        'XGBoost_prediction': y_hat_eight,
    })

    X = predictions_test[[
                        "RandomForest_Prediction",
                        'GBoost_prediction',
                        "XGBoost_prediction",
                        ]]

    X_interactions = poly.fit_transform(X)
    predictions = final_model.predict_proba(X_interactions)[:, 1]

    submission = pd.DataFrame({
        "ID": data_test["account.id"],
        "Predicted": predictions
    })

    submission.to_csv('predictions.csv', index=False)

X, y, features = feature_engineering(account=account)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state=0) ### i found restricting the model like this was more accurate a measure of what i would actually score

training_loop(X_train, y_train)
test_loop(X_test, y_test)

training_loop(X,y)

create_output(features=features)



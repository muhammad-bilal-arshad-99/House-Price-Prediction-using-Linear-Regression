import pandas as pd 
def load_data():
    df = pd.read_csv("housing.csv")
    return df

def clean_data(df):
    print(df.info())
    print(df.isnull().sum())

    # Option 1: Drop missing values
    df = df.dropna()

    # Option 2 (better for explanation):
    # df.fillna(df.mean(), inplace=True)

    return df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df[['area', 'bedrooms', 'bathrooms']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
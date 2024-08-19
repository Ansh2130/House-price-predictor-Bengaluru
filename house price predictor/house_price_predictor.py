#Program to predict house prices in banglore
#importing important library
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
#statemt to read the csv file
df1=pd.read_csv("bengaluru_house_prices.csv")
print(df1.to_string())
print(df1.shape) 
print(df1.groupby('area_type')['area_type'].agg('count'))
#removing unnecssary coluns from the data set
df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
#to check the null values in the data set
print(df2.head())
print(df2.isnull().sum())
#dropping all the rows with null values
df3=df2.dropna()
print(df3.isnull().sum())
#printing the size column in form of series
df3['size'].unique()
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0])) 
print(df3.head())
#as total sqft column may contain values in the form of ranges the fxn check wther the total sqft is single whole number or in the form of range
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
print(df3[~df3['total_sqft'].apply(is_float)].head())
#fxn to convert range to int and finding the avg of it
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head())
df5=df4.copy()
#creating a new column price per square feet
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
print(df5.head())
#removing unnecessary spaces from location column
df5.location=df5.location.apply(lambda x:x.strip())
location_stats=df5.groupby('location')['location'].agg('count')
print(location_stats)
#to print the length of how manyylocation are less than ten is mentioned in the data set
print(len(location_stats[location_stats<=10]))
location_stats_less_than_10=location_stats[location_stats<=10]
print(location_stats_less_than_10)
#replacing location whose count is less than ten with the word other
df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))


#PROCESS FOR REMOVING THE OUTLIERS



df6=df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)
#FXn depicts how toremove the outliers in case of priceper sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
print(df7.shape)
#plotting various graphs
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
    
plot_scatter_chart(df7,"Rajaji Nagar")
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
df8 = df7.copy()
print(df8.shape)
plot_scatter_chart(df8,"Rajaji Nagar")
plot_scatter_chart(df8,"Hebbal")
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()
df8.bath.unique()
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
plt.show()
print(df8[df8.bath>10])
df8[df8.bath>df8.bhk+2]
df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head(3))
#hot encoding for location
dummies = pd.get_dummies(df10.location)
print(dummies.head(3))
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print(df11.head())
df12 = df11.drop('location',axis='columns')
print(df12.head(2))
X = df12.drop(['price'],axis='columns')
print(X.head(3))
y = df12.price
print(y.head(3))
#Training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
print(lr_clf.score(X_test,y_test))
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print(cross_val_score(LinearRegression(), X, y, cv=cv))
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor


# Load the dataset

# Example: cross_val_score for LinearRegression
try:
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    lr_scores = cross_val_score(LinearRegression(), X, y, cv=cv)
    print("Cross-validation scores for Linear Regression:")
    print(lr_scores)
except Exception as e:
    print(f"Error performing cross-validation: {e}")
#grid-search method
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        try:
            gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
            gs.fit(X, y)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
            })
        except Exception as e:
            print(f"Error with {algo_name}: {e}")

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Find and print the best models
try:
    results = find_best_model_using_gridsearchcv(X, y)
    print("\nBest models and their parameters:")
    print(results)
except Exception as e:
    print(f"Error finding the best model: {e}")
def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = np.where(X.columns == location)[0][0]
    except IndexError:
        loc_index = -1

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

# Test the predict_price function
location = input("Enter the location:") 
sqft = int(input("Enter the sqft:") )
bath = int(input("Enter the no of bath:") )
bhk = int(input("Enter the bhk:") )
predicted_price = predict_price(location, sqft, bath, bhk)
print(f"The predicted price for a house with location {location}, {sqft} sqft, {bath} baths, and {bhk} BHK is: {predicted_price}")


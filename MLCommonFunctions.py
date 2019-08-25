#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This function performs dummy encoding for the data frame passed as an input parameter
def performDummyEncoding(df, categorical_features=None):
    if(categorical_features is None):
        categorical_features = df.dtypes[df.dtypes=="object"].index
    for counter, col_val in enumerate(categorical_features):
        tempDf = pd.get_dummies(df[col_val],drop_first = True,prefix=col_val + "_")
        # Add the results to the original housing dataframe
        df = pd.concat([df, tempDf], axis = 1)
        df = df.drop([col_val],axis=1)
    return df


# In[ ]:


#This function performs label encoding for the data frame passed as an input parameter
def performLabelEncoding(df, categorical_features=None):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    if(categorical_features is None):
        categorical_features = df.dtypes[df.dtypes=="object"].index
    for counter, col_val in enumerate(categorical_features):
        df[col_val]= label_encoder.fit_transform(df[col_val]) 
    return df


# In[ ]:


# plot the correlation matrix.
# it is based on the threshold on both the side
def plotCorrelationMatrix(df, positiveThreshold=0, negativeThreshold=-0.0):
    # get numeric features
    numericFeatures = df.dtypes[df.dtypes!="object"].index
    numericData = df[numericFeatures]
    corr = numericData.corr()
    dimension = numericData.shape[1]
    plt.figure(figsize=(dimension, dimension))
    #sns.heatmap(numericData.corr(), annot = True, cmap="YlGnBu")

    sns.heatmap(corr[(corr >= positiveThreshold) | (corr <= negativeThreshold)], 
            cmap='YlGnBu', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, square=True);


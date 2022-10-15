
# Change name 3




























# import numpy as np
# import pandas as pd

# from keras.models import Sequential
# from keras.layers import Dense

# # Loading the data

# CarPricesDataNumeric = pd.read_pickle("C:/Users/jose/Documents/Python code/DDBB/CarPricesData.pkl")


# target = ['Price']
# Predictors = ['Age', 'KM', 'Weight', 'HP', 'MetColor', 'CC', 'Doors']

# x = CarPricesDataNumeric[Predictors].values
# y = CarPricesDataNumeric[target].values



# # Standardization of data

# from sklearn.preprocessing import StandardScaler

# PredictorScaler = StandardScaler()
# TargetVarScaler = StandardScaler()

# # Storing the fit objects for later reference
# PredictorScalerFit = PredictorScaler.fit(x)
# TargetVarScalerFit = TargetVarScaler.fit(y)


# # Generating the standarized values of x and y
# x = PredictorScalerFit.transform(x)
# y = TargetVarScalerFit.transform(y)


# # Split the data into training and testing set
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)



# # =============================================================================
# # Creating the model ANN

# # With 3 hidden layers and 1 output that is the prediction price
# # =============================================================================


# # In the below code snippet, the “Sequential” module from the Keras library 
# # is used to create a sequence of ANN layers stacked one after the other. 
# # Each layer is defined using the “Dense” module of Keras where we specify how 
# # many neurons would be there, which technique would be used to initialize the 
# # weights in the network. what will be the activation function for each neuron 
# # in that layer etc

# from keras.models import Sequential
# from keras.layers import Dense

# # Create ANN model

# model = Sequential()

# # Defining the input layer and first hidden layer, both are the same!
# model.add(Dense(units = 5, input_dim = 7, kernel_initializer='normal', activation='relu'))

# # Defing the second layer of the model
# model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))

# # The output neuron is a single fully connected node
# # since we will be predicting a single number
# model.add(Dense(1, kernel_initializer='normal'))

# # Compiling the model
# model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# # Fitting the ANN to the training set
# model.fit(x_train, y_train, batch_size = 20, epochs=50, verbose=1)







# # # =============================================================================
# # # Hyperparameters tunning
# # # =============================================================================

# # # Using manual grid search



# # # Defining a function to find the best parameters for ANN
# # def functionFindBestParameters(x_train, y_train, x_test, y_test):
    
# #     # Defining the list of hyperparameters to try
# #     batch_size_list = [5, 10, 15, 20]
# #     epoch_list = [5, 10, 50, 100]
    
# #     searchResultData = pd.DataFrame(columns = ['TrialNumer', 'Parameters', 'Accuracy'])
    
# #     # Initialize the trials
# #     trialNumber = 0
# #     for batch_size_trial in batch_size_list:
# #         for epoch_trial in epoch_list:
# #             trialNumber += 1
            
            
# #             # batch_size_trial = 5
# #             # epoch_trial = 5
            
# #             # Create ANN Model
# #             model = Sequential()
            
# #             # Defining the first layer of the model
# #             model.add(Dense(units = 5, input_dim = x_train.shape[1], kernel_initializer='normal', activation='relu'))
            
# #             # Defining the second layer of the model
# #             model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
            
# #             # The output neuron is a single fully conected node
# #             # Since we will be predicting a single number
# #             model.add(Dense(1, kernel_initializer='normal'))
            
# #             # Compiling the model
# #             model.compile(loss='mean_squared_error', optimizer='adam')
            
# #             # Fitting the ANN the training set
# #             model.fit(x_train, y_train, batch_size=batch_size_trial, epochs=epoch_trial, verbose=0)
# #             model.fit(x_train, y_train, batch_size=5, epochs=5, verbose=0)
            
# #             MAPE = np.mean(100 * (np.abs(y_test - model.predict(x_test))/y_test))
            
# #             # Printing the results of the current iteration
# #             print(trialNumber, 'Parameters:', 'batch_size_trail:', batch_size_trial,'-', 'epochs:', epoch_trial, 'Accuracy:', 100 - MAPE)

            
# #             searchResultData = searchResultData.append(pd.DataFrame(data = [[trialNumber, str(batch_size_trial)+'-'+str(epoch_trial), 100-MAPE]],
# #                                                                     columns = ['Trial Number', 'Parameters', 'Accuracy']))
            
# #     return(searchResultData)

# # # Calling the function
# # ResultsData = functionFindBestParameters(x_train, y_train, x_test, y_test)


# # # =============================================================================
# # # Plotting the parameters trial results            
# # # =============================================================================
    
# # ResultsData.plot(x = 'Parameters', y = 'Accuracy', figsize = (15, 4), kind = 'line')



# # # =============================================================================
# # # Trainig the ANN model with the best parameters
# # # =============================================================================


# # # Fitting the ANN to the training set
# # model.fit(x_train, y_train, batch_size=15, epochs = 5, verbose = 0)

# # # Generating predictions on testing data
# # predictions = model.predict(x_test)

# # # Scaling the predicted Price data back to original price scale
# # predictions = TargetVarScalerFit.inverse_transform(predictions)

# # # Scaling the y_test price data back to original price scale
# # y_test_orig = TargetVarScalerFit.inverse_transform(y_test)

# # # Scaling the test data back to original scale
# # test_data = PredictorScalerFit.inverse_transform(x_test)

# # testing_data = pd.DataFrame(data = test_data, columns = Predictors)
# # testing_data['Price'] = y_test_orig
# # testing_data['PredictedPrice'] = predictions
# # testing_data.head()


# # # Finding the accuracy of the model

# # # Computing the absolute percent error
# # APE = 100 * (abs(testing_data['Price'] - testing_data['PredictedPrice']) / testing_data['Price'])
# # testing_data['APE'] = APE

# # print("The accuracy of the model is: ", 100 - np.mean(APE))





# # =============================================================================
# # Finding the best parameters using GridSearchCV
# # =============================================================================

# # Function to generate deep ANN model

# from keras.models import Sequential
# from keras.layers import Dense

# def make_regression_ann(optimizer_trail):
#     model = Sequential()
#     model.add(Dense(units = 5, input_dim = 7, kernel_initializer='normal', activation = 'relu'))
#     model.add(Dense(units = 5, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mean_squared_error', optimizer = optimizer_trail)
#     return model

# ###################################################
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasRegressor    

# # List all the parameters to try
# parameters_trail = {'batch_size' : [10, 20, 30],
#                     'epochs' : [10, 20],
#                     'optimizer_trail' : ['adam', 'rmsprop']}

# # Creating the regression ANN model
# regModel = KerasRegressor(make_regression_ann, verbose = 0)


# from sklearn.metrics import make_scorer

# # Defining a custom function to calculate accuracy
# def accuracy_score(orig, pred):
#     MAPE = np.mean(100*(np.abs(orig - pred)/orig))
#     print('#'*70, 'Accuracy:', 100 - MAPE)
#     return(100 - MAPE)

# custom_scoring = make_scorer(accuracy_score, greater_is_better=True)


# # Creating the grid seach space
# # See different scoring methods by using sklearn.metrics.SCORES.keys()

# grid_search = GridSearchCV(
#     estimator=regModel,
#     param_grid=parameters_trail,
#     scoring = custom_scoring,
#     cv = 5
#     )

# # Measuring how much time it takes to find the parameters
# import time
# startTime = time.time()

# # Running grid search for different parameters
# grid_search.fit(x, y, verbose = 1)

# endTime = time.time()
# print("### Total time taken: ", round((endTime - startTime)/60), 'Minutes')

# print("## Best parameters")
# grid_search.best_params_





























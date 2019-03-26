from io import StringIO
import tensorflow as tf
import numpy as np
import requests

from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify, json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import pymysql
import re
import datetime
import matplotlib.pyplot as plt
import ann_tflearn

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas import Series
# from nlp import chatbot

# from ann import NeuralNetwork
from barchart import barchart

app = Flask("__name__")

@app.route('/')
def red():
    return render_template('main.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/PredictionInput')
def PredictionInput():
    return render_template('PredictionInput.html')

@app.route('/fetchapi',methods=['POST', 'GET'])
def fetchapi():
    now = datetime.datetime.now()
    date = request.form.get('date')

    print("date")
    print(date)


    # print("date")
    # print(date)
    # currentdate =now.strftime("%Y-%m-%d")
    # print(currentdate)
    # parameters = {"lat": 40.71, "lon": -74}
    # # response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
    # # response = requests.get("http://api.openweathermap.org/data/2.5/weather?lat=145.77&lon=-16.92")
    #
    # # for current date
    # response = requests.get("https://api.ipdata.co/?api-key=test")s
    # json_data = json.loads(response.content)
    # print(json_data)
    #
    # lat=str(json_data['latitude'])
    # print(lat)
    # lon = str(json_data['longitude'])
    # print(lon)
    #
    #
    # if(date == currentdate or date < currentdate ):
    #     print("same")
    #     response = requests.get(" https://samples.openweathermap.org/data/2.5/weather?lat="+lat+"&lon="+lon+"&appid=b6907d289e10d714a6e88b30761fae22")
    #     json_data = json.loads(response.content)
    #     print(json_data)
    #     print(json_data['wind']['speed'])
    #
    #     return jsonify(json_data)
    # if(date > currentdate):
    #     response = requests.get("https://samples.openweathermap.org/data/2.5/forecast/daily?lat="+lat+"&lon="+lon+"&cnt=1&appid=b1b15e88fa797225412429c1c50c122a1")
    #     print("future")
    #     json_data = json.loads(response.content)
    #     print(json_data)
    con = pymysql.connect(host="localhost", user="root", password="", database="solarprediction")
    cur = con.cursor()
    sql = "select * from previous_record where date=%s"
    val = (date)
    cur.execute(sql, val)
    rows = cur.fetchall();
    tempList = []


    for row in rows:
        print("hiiiiiiiiiiiiiihello")
        print(row)
        tempList.append(row)
    cur.close()
    return jsonify(tempList)
    # rowLen = len(rows)
    # if (rowLen > 0):
    #     print("full")
    #     return render_template("userPrediction.jsp")
    #
    # else:
    #     print("empty")
    #     return render_template("UserLogin.html", message="Enter correct username and password")









@app.route('/Prediction', methods=['POST', 'GET'])
def Prediction():
    predicValue = request.form['predictAlgo']
    print("predicValue" + predicValue)
    if(predicValue == 'SVM'):
        prediction =svmPredict()
        return render_template('predictionResult.html',prediction=prediction)
    elif(predicValue == 'ANN'):
        print("ANN")
        prediction = annPredict()
        return render_template('predictionResult.html', prediction=prediction)
    elif (predicValue == 'Linear'):
        print("Linear")
        prediction = LinearPredict()
        return render_template('predictionResult.html', prediction=prediction)
    elif (predicValue == 'Logistic'):
        print("Logistic")
        prediction = LogisticPredict()
        return render_template('predictionResult.html',prediction=prediction)
    elif (predicValue =='errorGraph'):
        relativeErrorGraph()
    elif (predicValue == 'predictGraph'):
        predictGraph()



def svmPredict():


    TEMPERATURE = request.form['TEMPERATURE']
    ATMOSPHERE = request.form['ATMOSPHERE']
    HUMIDITY = request.form['HUMIDITY']
    WINDSPEED = request.form['WINDSPEED']
    SUNSHINE = request.form['SUNSHINE']
    AMOUNTOFCLOUD = request.form['AMOUNTOFCLOUD']
    GLOBALSOLARRADIATION = request.form['GLOBALSOLARRADIATION']
    DIFFUSESOLARRADIATION = request.form['DIFFUSESOLARRADIATION']

    print("rummmmmm")
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.corr())
    # df = (df - df.mean()) / (df.max() - df.min())

    df.replace('?',-99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    print(df.head())
    df.drop(['Date'],1, inplace=True)  # drop date column


    X = np.array(df.drop(['Solarenergy'], 1))

    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    # y=np.array([0 if elem < 3000 else 1 for elem in df['Solar energy']])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # clf = svm.SVC()
    #
    # clf.fit(X_train, y_train)

    svclassifier = svm.SVR(kernel='linear')

    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))

    print(type(y_test))
    print(type(y_pred))

    print(y_test.shape)
    print(y_pred.shape)

    accuracy = r2_score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
    print(accuracy)
    #confidence = svclassifier.score(X_test, y_test)
    #print(confidence)

    example_measures = np.array([[TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    prediction = svclassifier.predict(example_measures)

    x_axis = [i+1 for i in range(y_pred.shape[0])]
    plt.plot(x_axis, y_pred, 'b')
    plt.plot(x_axis, y_test, 'r')
    plt.show()

   # print(prediction)
#    print('Accuracy', metrics.accuracy_score(y_test, y_pred))
    #print('Mean Absolute Error: SVM', metrics.mean_absolute_error(y_test, y_pred))
    # return render_template('predictionResult.html', prediction=prediction)
    return prediction

def annPredict():
    TEMPERATURE = float(request.form['TEMPERATURE'])
    ATMOSPHERE = float(request.form['ATMOSPHERE'])
    HUMIDITY = float(request.form['HUMIDITY'])
    WINDSPEED = float(request.form['WINDSPEED'])
    SUNSHINE = float(request.form['SUNSHINE'])
    AMOUNTOFCLOUD = float(request.form['AMOUNTOFCLOUD'])
    GLOBALSOLARRADIATION = float(request.form['GLOBALSOLARRADIATION'])
    DIFFUSESOLARRADIATION = float(request.form['DIFFUSESOLARRADIATION'])

    # nn = NeuralNetwork([8, 8, 1])
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))
    np.append([TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION], 1)
    y = np.array(df['Solarenergy'])

    y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # ann_tflearn.ann(X_train, X_test, y_train, y_test)

    mlp = MLPRegressor(solver='adam', activation='relu', alpha=3e-3, hidden_layer_sizes=(512, 1), max_iter=100)
    
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
	# y_pred = y_pred.reshape(-1, 1)
    print("shape of x test is:", X_test.shape)
    print(type(X_test))
    print(y_pred.shape)
    print(y_test.shape)
    accuracy = r2_score(y_pred.reshape(-1, 1), y_test)
    print("#######################################")
    for i in range(len(list(y_pred))):
        print(list(y_test)[i], ":", list(y_pred)[i])
    print("#######################################")
    # nn.fit(X_train, y_train)
    a=''
    # y_predict = []
    # for e in list(X_test):
    #     y_predict.append(nn.predict(e))
        # a = str(e)+str(nn.predict(e))+"\n"
    # accuracy = r2_score(list(y_pred), list(y_test))
    print(accuracy)


    input_data = np.array([TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION])
    print(input_data.shape)
    input_data = input_data.reshape(1, -1)
    print(input_data.shape)
    print(type(input_data))
    print(input_data[0])
    prediction = mlp.predict(input_data[0].reshape(1, -1))
    return prediction


def LinearPredict():
    TEMPERATURE = request.form['TEMPERATURE']
    ATMOSPHERE = request.form['ATMOSPHERE']
    HUMIDITY = request.form['HUMIDITY']
    WINDSPEED = request.form['WINDSPEED']
    SUNSHINE = request.form['SUNSHINE']
    AMOUNTOFCLOUD = request.form['AMOUNTOFCLOUD']
    GLOBALSOLARRADIATION = request.form['GLOBALSOLARRADIATION']
    DIFFUSESOLARRADIATION = request.form['DIFFUSESOLARRADIATION']
    print("rummmmmm")
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))

    np.append([TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION],1)
    # y = np.array(df['Class'])
    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print(X_test)
    print("X_test");
    print(X_test);
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("1111");
    print(regressor.intercept_)
    print("222222222");
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)
    print(y_pred)
    accuracy = r2_score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
    print('Accuracy : Linear Regression',accuracy)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df
    print('Mean Absolute Error: Linear', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return y_pred[-1]



def LogisticPredict():
    TEMPERATURE = float(request.form['TEMPERATURE'])
    ATMOSPHERE = float(request.form['ATMOSPHERE'])
    HUMIDITY = float(request.form['HUMIDITY'])
    WINDSPEED = float(request.form['WINDSPEED'])
    SUNSHINE = float(request.form['SUNSHINE'])
    AMOUNTOFCLOUD = float(request.form['AMOUNTOFCLOUD'])
    GLOBALSOLARRADIATION = float(request.form['GLOBALSOLARRADIATION'])
    DIFFUSESOLARRADIATION = float(request.form['DIFFUSESOLARRADIATION'])

    print("rummmmmm")
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))
    np.append([TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION], 1)
    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    # y = np.array([0 if elem < 3000 else 1 for elem in df['Solar energy']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Mean Absolute Error: Logistic ', metrics.mean_absolute_error(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(y_pred)
    accuracy = r2_score(y_pred.reshape(-1, 1), y_test)
    print(accuracy)
    for i in range(len(list(y_pred))):
        print(list(y_test)[i], ":", list(y_pred)[i])

    input_data = np.array([TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION])
    input_data = input_data.reshape(1, -1)
    prediction = logreg.predict(input_data)
    return prediction

@app.route('/User')
def user():
    return render_template('UserLogin.html')

@app.route('/userlogin', methods=['POST', 'GET'])
def userlogin():
    username = request.form['username']
    password = request.form['password']

    con = pymysql.connect(host="localhost", user="root", password="", database="solarprediction")
    cur = con.cursor()
    sql = "select * from user_table where email=%s and password=%s"
    val = (username, password)
    cur.execute(sql, val)
    rows = cur.fetchall();
    rowLen = len(rows)
    if (rowLen > 0):
        print("full")
        return render_template("userPrediction.jsp")

    else:
        print("empty")
        return render_template("UserLogin.html", message="Enter correct username and password")


@app.route('/adminlogin', methods=['POST', 'GET'])
def adminlogin():
    print("ppppppppppppppp11")
    username = request.form['username']
    password = request.form['password']

    con = pymysql.connect(host="localhost", user="root", password="", database="solarprediction")
    cur = con.cursor()
    sql= "select * from admin_table where email=%s and password=%s"
    val=(username, password)
    cur.execute(sql, val)
    rows =cur.fetchall();
    rowLen =len(rows)
    if(rowLen > 0):
        print("full")
        return render_template("AdminMain.html")

    else:
        print("empty")
        return render_template("Login.html", message="Enter correct username and password")



@app.route('/Admin')
def Admin():
    return render_template("Login.html")

@app.route('/addQuestion',methods=['POST', 'GET'])
def addQuestion():
    print("pooooop")

    question = request.form['question']
    answer = request.form['answer']
    con = pymysql.connect(host="localhost", user="root", password="", database="solarprediction")
    cur = con.cursor()
    sql = "insert into question_table(question,answer) values(%s,%s)"
    values=(question, answer)
    cur.execute(sql, values)
    con.commit()
    return render_template("addQuestion.html", msg="Questions added successfully")

@app.route("/chatBox")
def chatBox():
    return render_template("chatbox.html")

@app.route("/fetchAnswers",methods=['POST','GET'])
def fetchAnswers():
    question = request.form['msg']
    response= chatbot.userresponse(question)


    # question =' '.join(question.split())
    # print("fetchAnswer"+question)
    # questionLen =len(question.split(" "))
    # print("question length")
    # print(questionLen)
    # wordsList = question.split(" ")
    #
    # for words in wordsList :
    #     print("wwwwwww "+words)
    #
    #
    #
    #
    # con = pymysql.connect(host="localhost", user="root", password="", database="solarprediction")
    # cur = con.cursor()
    # sql = "select * from question_table"
    # cur.execute(sql)
    # rows = cur.fetchall()
    # for row in rows:
    #    print(row)
    #    print(row[1])
    #
    #    count = 0
    #    databaseCount = len(row[1].split(" "))
    #    print("databaseCount")
    #    print(databaseCount)
    #    for words in wordsList:
    #        print("wwwwwww " + words)
    #
    #        if words in row[1]:
    #
    #          count = count + 1
    #    finalcount = count
    #    print(finalcount)
    #    if finalcount >= databaseCount/2 :
    #      print("greater")
    #      return row[2]
    #    else:
    #      print("smaller")
    return response


@app.route("/relativeErrorGraph",methods=['POST','GET'])
def relativeErrorGraph():
    print("Relative Error graph")
    TEMPERATURE = request.form['TEMPERATURE']
    ATMOSPHERE = request.form['ATMOSPHERE']
    HUMIDITY = request.form['HUMIDITY']
    WINDSPEED = request.form['WINDSPEED']
    SUNSHINE = request.form['SUNSHINE']
    AMOUNTOFCLOUD = request.form['AMOUNTOFCLOUD']
    GLOBALSOLARRADIATION = request.form['GLOBALSOLARRADIATION']
    DIFFUSESOLARRADIATION = request.form['DIFFUSESOLARRADIATION']
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))
    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    # y=np.array([0 if elem < 3000 else 1 for elem in df['Solar energy']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # clf = svm.SVC()
    #
    # clf.fit(X_train, y_train)

    svclassifier = svm.SVC(kernel='sigmoid')

    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    confidence = svclassifier.score(X_test, y_test)
    print(confidence)

    example_measures = np.array(
        [[TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    prediction = svclassifier.predict(example_measures)
    print(prediction)
    print('Mean Absolute Error: SVM', metrics.mean_absolute_error(y_test, y_pred))
    svmError = metrics.mean_absolute_error(y_test, y_pred);

    nn = NeuralNetwork([8, 8, 1])
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))
    np.append(
        [TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION], 1)
    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    print('Mean Absolute Error: ANN', metrics.mean_absolute_error(y_test, y_pred))
    annError =  metrics.mean_absolute_error(y_test, y_pred)

    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))

    np.append(
        [TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION], 1)
    # y = np.array(df['Class'])
    y = np.array(df['Solarenergy'])
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    print(X_test)

    print("X_test");
    print(X_test);
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("1111");
    print(regressor.intercept_)
    print("222222222");
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)
    print(y_pred)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df
    print('Mean Absolute Error: Linear', metrics.mean_absolute_error(y_test, y_pred))

    linearError = metrics.mean_absolute_error(y_test, y_pred)

    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))
    np.append(
        [TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION], 1)
    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    # y = np.array([0 if elem < 3000 else 1 for elem in df['Solar energy']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Mean Absolute Error: Logistic ', metrics.mean_absolute_error(y_test, y_pred))


    logisticError = metrics.mean_absolute_error(y_test, y_pred)

    objects = ('SVM', 'ANN', 'LINEAR ', 'LOGISTIC')
    y_pos = np.arange(len(objects))
    performance = [svmError, annError, linearError, logisticError]

    plt.bar(y_pos, performance, align='center', alpha=1)
    plt.xticks(y_pos, objects)
    plt.ylabel('Error')
    plt.title('Relative Error')
    plt.show()

@app.route('/predictGraph',methods=['POST','GET'])
def predictGraph():
    TEMPERATURE = request.form['TEMPERATURE']
    ATMOSPHERE = request.form['ATMOSPHERE']
    HUMIDITY = request.form['HUMIDITY']
    WINDSPEED = request.form['WINDSPEED']
    SUNSHINE = request.form['SUNSHINE']
    AMOUNTOFCLOUD = request.form['AMOUNTOFCLOUD']
    GLOBALSOLARRADIATION = request.form['GLOBALSOLARRADIATION']
    DIFFUSESOLARRADIATION = request.form['DIFFUSESOLARRADIATION']
    df = pd.read_csv('E:/Mumbai-solar.csv')
    df.shape
    df.head()
    print(df.head())
    df.replace('?', -99999, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(['Date'], 1, inplace=True)  # drop date column

    X = np.array(df.drop(['Solarenergy'], 1))
    y = np.array(df['Solarenergy'])
    y = y.astype('int')
    # y=np.array([0 if elem < 3000 else 1 for elem in df['Solar energy']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # clf = svm.SVC()
    #
    # clf.fit(X_train, y_train)

    svclassifier = svm.SVC(kernel='sigmoid')

    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    confidence = svclassifier.score(X_test, y_test)
    print(confidence)

    example_measures = np.array(
        [[TEMPERATURE,ATMOSPHERE,HUMIDITY,WINDSPEED,SUNSHINE,AMOUNTOFCLOUD,GLOBALSOLARRADIATION,DIFFUSESOLARRADIATION]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    prediction = svclassifier.predict(example_measures)
    print(prediction)
    print('Mean Absolute Error: SVM', metrics.mean_absolute_error(y_test, y_pred))
    # return render_template('predictionResult.html', prediction=prediction)
    svmpredict = prediction

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("1111");
    print(regressor.intercept_)
    print("222222222");
    print(regressor.coef_)
    y_pred = regressor.predict(X_test)
    print(y_pred)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df
    print('Mean Absolute Error: Linear', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    linearPredict = y_pred[-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Mean Absolute Error: Logistic ', metrics.mean_absolute_error(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(y_pred)
    logisticPredict = y_pred[-1]

    objects = ('SVM', 'LINEAR ', 'LOGISTIC')
    y_pos = np.arange(len(objects))
    performance = [svmpredict,  linearPredict, logisticPredict]

    plt.bar(y_pos, performance, align='center', alpha=1)
    plt.xticks(y_pos, objects)
    plt.ylabel('Error')
    plt.title('Relative Error')
    plt.show()


if __name__ == '__main__':
    app.run(debug=True)

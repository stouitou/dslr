V.2 Data visualization :
    V. 2.1 Histogram :
        histogram.py ->
            Which Hogwarts course has a homogeneous score distribution between all four houses?
    V. 2.2 Scatter plot :
        scatter_plot.py ->
            What are the two features that are similar?

V. 3 Logistic Regression :
    One-vs-all (one-vs-rest)
    logreg_train.py (dataset_train.csv as parameter) ->
        - uses gradient descent to minimize the error
        - generates a file containing the weights for prediction
    logreg_predict.py (dataset_test.csv and the file containing the weights as parameter) ->
        - generates a prediction file houses.csv :
            Index,Hogwarts House
            0,Gryffindor
            1,Hufflepuff
            2,Ravenclaw
            3,Hufflepuff
            4,Slytherin
            5,Ravenclaw
            6,Hufflepuf
                [...]

Your classifier will be evaluated on the data present in dataset_test.csv. Your answers will be evaluated using the accuracy score from the Scikit-Learn library. Professor McGonagall agrees that your algorithm is comparable to the Sorting Hat only if it has a minimum accuracy score of 98%.

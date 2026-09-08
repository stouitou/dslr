V.1 Data analysis :
    describe.py (dataset as parameter) ->
        - count
        - mean
        - std
        - min
        - 25% (percentiles)
        - 50% (percentiles)
        - 75% (percentiles)
        - max

    Se partager ce qui est a renvoyer ?
    Preparer les donnees, "annuler" les valeurs nulles (ne pas compter / compter comme moyenne ?), normaliser

V.2 Data visualization :
    V. 2.1 Histogram :
        histogram.py ->
            Which Hogwarts course has a homogeneous score distribution between all four houses?
    V. 2.2 Scatter plot :
        scatter_plot.py ->
            What are the two features that are similar?
    V. 2.3 Pair plot :
        pair_plot.py ->
            (or scatter plot matrix)
            From this visualization, which features are you going to use for your logistic regression?
    
    Un commence par l'histogram, l'autre par le scatter plot. Celui qui finit en premier fait le dernier ?

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

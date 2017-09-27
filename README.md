# Udacity_Machine_Learning_Nanodegree
The Machine Learning Engineer Nanodegree in Udacity
- **Titanic Survival Exploration**
    - In this project, decision functions are created that attempt to predict survival outcomes from the 1912 Titanic disaster based on each passenger’s features, such as sex and age.
    - A simple algorithm is tested to accurately predict the outcomes for at least 80% of the passengers in the provided data.
    
- **Boston Housing**
    - In this project, a _DecisionTreeRegressor_ is used to predict the Boston housing price, as a _supervised learning_ and _regression_ problem.
    - The `r2_score` is used as metrics.
    - Model performance, including _bias-variance tradeoff_, is analyzed by examing the learning curves (as a function of iterations) and complexity curves (as a function of model hyper-parameter).
    - Hyper-parameters are chosen by grid search.
    
- **Finding Donors for CharityML**
    - The goal of this project is to evaluate and optimize several different _supervised learners_ to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent (_classification problem_).
    - Data preprocessing includes  _log-transform_ (due to the skewness of data), `MinMaxScaler`, and `get_dummies`.
    - The _accuracy_, _precision_, _recall_, _F-beta score_, and _confusion_matrix_ are used for evaluation.
    - Several classifiers are testes, such as `LogisticREgression`,  `KNeighborsClassifier`, `GaussianNB`, `svm`, `SDGClassifier`, `RandomForestClassifier`, `AdaBoostClassifer`, `GradientBoostingClassifier`, where `AdaBoostClassifier` is the model with the best performance.
    
- **Creating Customer Segments**
    - This project is to use _unsupervised learning_ techniques to see if any similarities exist between customers, and how to best segment customers into distinct categories.
    - The `scatter_matrix` and `corr` are used to visualize the relevance between features.
    - The `boxcox` and _log_transformation_ are used for data transformation.
    - Outliers are deceted using _IQR_.
    - _Principal component analysis (PCA)_ is used for data visualization in biplot.
    - The _k_means_ and `GaussianMixture` models are used for clustering, with `silhouette_score` to determine the  optimal clustering number.
    
- **Train a Smartcab to Drive**
   - This project is to use _reinforcement learning techniques_ to construct a demonstration of a smartcab operating in real-time to prove that both safety and efficiency can be achieved.
   - Q-learning is used in forms of  `Q(s(t), a(t)) := Q(s(t), a(t)) + learning_rate * (reward(t) + gamma * max Q(s(t+1), a) - Q(s(t), a(t))`, where gamma is not presented.
   
- **Dog Breed Classifier**
    - In this project, a pipeline is built to process real-world, user-supplied images. Given an image of a dog, the algorithm used _Convolutional Neural Network (CNN)_ will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.
    - The `keras` is used and `tensorflow` is the backend for CNN implementation.
    - To reduce training time, _transfer learning_ is used, leveraging on the pre-trained VGG-16 model as a fixed feature extractor.
    
- **Chicago Crimes**
    - In this project, the Chicago crime data is analyized and categorized into crimes types given time, location and solution, using machine learning algorithms. Data is imported from Chicago Data Portal website.
    - _Logistic regression_ is used as a benchmark model and _XGBoost_ algorithm is applied with created new features and optimized parameters.

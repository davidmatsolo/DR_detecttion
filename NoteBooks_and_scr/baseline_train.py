import sys
sys.path.append('D:/Advanced AI/Project/')

from model_archtectures.BaselineModel import BaselineModel
from model_archtectures.NaiveModel import NaiveModel

if __name__ == "__main__":

    
    baseline_model = BaselineModel()
    directory_path = 'D:/Advanced AI/Project/colored_images/'
    
    classes = {'No_DR': 0, 'Moderate': 1, 'Severe': 2}
    X, Y = baseline_model.get_training_data(directory_path, classes)
    #baseline_model.plot_sample_images(X, Y)
    X_processed = baseline_model.preprocess_data(X)
    xtrain, xvalid, ytrain, yvalid = baseline_model.split_data(X_processed, Y)
    baseline_model.train_and_evaluate_svc_model(xtrain, ytrain, xvalid, yvalid)

    print("Testing")
    directory_path = 'D:/Advanced AI/Project/colored_images/Testing/'
    xtest, y_test = baseline_model.get_training_data(directory_path, classes)
    x_test_prep = X_processed = baseline_model.preprocess_data(xtest)
    baseline_model.view_metrics(x_test_prep,y_test) 



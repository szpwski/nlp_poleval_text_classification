"""
Module containing functions performing cross-validation and scoring results
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_validate
from .pytorch_dataset import Dataset
import torch

def perform_cross_validation(X, y, estimator, param_grid):
    scoring = {'accuracy': 'accuracy', 'precision':'precision', 'recall': 'recall', 'f1':'f1'}

    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, refit='f1', cv=3, return_train_score=True)
    #results = cross_validate(grid_search, X, y, scoring=scoring, cv=3, n_jobs = -1)

    # Obtain the best model
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    results = grid_search.cv_results_

    # # Print the results
    # print("Cross-validation results:")
    # for metric, scores in results.items():
    #     print(f"{metric}: {scores.mean()} (±{scores.std()})")

    return best_model, results, grid_search

def score(y_true, y_pred):
    """
    Function returns scores of predictions
    """
    return {
        'accuracy' : accuracy_score(y_true, y_pred),
        'precision' : precision_score(y_true, y_pred),
        'recall' : recall_score(y_true, y_pred),
        'f1' : f1_score(y_true, y_pred)
    }

def create_result_df(notes, labels, preds, step, epoch, scores):
    """
    Function return dataframe containing result predictions together with scores
    """
    
    results = pd.DataFrame({
        "notes" : notes,
        "labels" : labels,
        "predictions" : preds
    })
    
    results["step"] = step
    results["epoch"] = epoch
    
    for k in scores.keys():
        results[k] = scores.get(k)
    
    return results

def get_prob_scores(model, df_test):
    test = Dataset(df_test)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    labels, preds, prob_scores = torch.tensor([]), torch.tensor([]), torch.tensor([])

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            
            test_label = test_label.to(device)
            
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)

            output_classes = torch.tensor([1 if prob >= 0.5 else 0 for prob in output]).float().to(device)
            
            labels = torch.cat((labels, test_label.to("cpu")))
            preds = torch.cat((preds, output_classes.to("cpu")))
            
    y_true = labels.numpy()
    y_prob = preds.detach().numpy()
    y_pred = preds.round().detach().numpy()

    return y_true, y_prob, y_pred


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model


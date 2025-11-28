Moin!

File "main.py" is the initial Python project in which I developed the code.
File "main.ipynb" is a more beatiful version of it in JupyterNotebook with workflow explanations.

- You can find detailed summary of the results at the end of JupyterNotebook.
- It also contains answers to questions from assignment description.

## Overall result ##
XGBoostClassifier showed the best performance among the other models. It has been chosen as the final model.
Our primary goal is to maximize lead coverage (Recall) because the cost of a False Negative (a potential subscriber we fail to contact) is a lost revenue opportunity that significantly outweighs the wasted effort of a False Positive (a non-subscriber we contact).
- ROC-AUC-Score: 0.80
- Recall: 0.75
- Precision: 0.26
- F1-score: 0.37

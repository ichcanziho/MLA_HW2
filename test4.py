from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import roc_curve

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]
    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    ns_probs = [0 for _ in range(len(new_actual_class))]
    ns_auc = roc_auc_score(new_actual_class, ns_probs)

    ns_fpr, ns_tpr, _ = roc_curve(new_actual_class, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(new_actual_class, new_pred_class)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')


    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate Class:'+str(per_class))
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


    roc_auc_dict[per_class] = roc_auc
    avg = sum(roc_auc_dict.values())/len(roc_auc_dict)
  return roc_auc_dict,avg

print("\nLogistic Regression")
actual_class = [0,0,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,1]
predicted_class=[0,1,1,0,1,1,0,1,1,0,0,0,0,1,1,1,1,1]
# assuming your already have a list of actual_class and predicted_class from the logistic regression classifier
lr_roc_auc_multiclass,avg = roc_auc_score_multiclass(actual_class, predicted_class)
print(lr_roc_auc_multiclass,avg)
import numpy as np
y_true =   np.array([0,0,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,1])
y_scores = np.array([0,1,1,0,1,1,0,1,1,0,0,0,0,1,1,1,1,1])
r=roc_auc_score(y_true, y_scores)
print(r)
ns_probs = [0 for _ in range(len(y_true))]
ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_true, y_scores)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate Class:'+str(3))
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import roc_curve


def plotCurve(new_actual_class, new_pred_class, current_class, model_name, multi=False):
    ns_probs = [0 for _ in range(len(new_actual_class))]
    ns_fpr, ns_tpr, _ = roc_curve(new_actual_class, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(new_actual_class, new_pred_class)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label=model_name)
    # axis labels
    classes = set(new_actual_class)
    if multi:
        pyplot.xlabel('False Positive Rate Class:' + str(current_class))
    else:
        pyplot.xlabel('False Positive Rate')

    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()


def roc_auc_score_multiclass(y_true, y_pred, model_name, curve=False, average="macro"):
    classes = set(y_true)
    roc_auc_dict = {}
    for current_class in classes:
        # creating a list of all the classes except the current class
        other_class = [x for x in classes if x != current_class]
        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in y_true]
        new_pred_class = [0 if x in other_class else 1 for x in y_pred]
        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[current_class] = roc_auc
        if curve:
            if len(classes) == 2 and current_class == 1:
                plotCurve(new_actual_class, new_pred_class, current_class, model_name)
            if len(classes) != 2:
                plotCurve(new_actual_class, new_pred_class, current_class, model_name, multi=True)

    avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)
    return avg, roc_auc_dict

if __name__ == "__main__":
    actual_class = [0,0,1,1,1,1,1,0,0]
    predicted_class = [0,1,1,1,1,1,1,0,0]
    auc_avg, roc_auc_multiclass = roc_auc_score_multiclass(actual_class, predicted_class, "SVM", curve=True)
    print(roc_auc_multiclass, auc_avg)

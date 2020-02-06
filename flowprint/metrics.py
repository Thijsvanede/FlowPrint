from sklearn.metrics import confusion_matrix
import numpy as np

class metrics(object):

    def tfpn(self, y_true, y_pred):
        """Compute True/False Positive/Negative values."""
        # Transform to numpy array
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Compute True/False Positives/Negatives
        true_positives = np.diag(cm)
        true_negatives = np.zeros(cm.shape[0])
        false_positives = np.sum(cm, axis=0) - true_positives
        false_negatives = np.sum(cm, axis=1) - true_positives

        # Return result
        return true_positives, true_negatives, false_positives, false_negatives


    def compute(self, y_true, y_pred):
        """"""
        # Compute True/False Positives/Negatives
        tp, tn, fp, fn = self.tfpn(y_true, y_pred)

        # Compute classification report
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f1_score  = 2*tp / (2*tp + fp + fn)
        accuracy  = (tp + tn) / (tp + tn + fp + fn)
        support   = np.array([np.sum(y_true == y) for y in np.unique(y_true)])

        # Set nan values to 0
        precision[np.isnan(precision)] = 0
        recall   [np.isnan(recall   )] = 0
        f1_score [np.isnan(f1_score )] = 0
        accuracy [np.isnan(accuracy )] = 0

        # Compute average values
        precision_avg = np.sum(precision * support) / np.sum(support)
        recall_avg    = np.sum(recall    * support) / np.sum(support)
        f1_score_avg  = 2*precision_avg*recall_avg / (precision_avg + recall_avg)
        accuracy_avg  = np.sum(accuracy  * support) / np.sum(support)
        print("Precision:", precision_avg)
        print("Recall   :", recall_avg)
        print("F1-score :", f1_score_avg)
        print("Accuracy :", accuracy_avg)
        print("Support  :", support)

        from sklearn.metrics import accuracy_score
        print("Accuracy1: ", accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        print("Accuracy2: ", np.diag(cm).sum()/cm.sum())

        print("Accuracy3: ", (tp.sum()+tn.sum())/(tp.sum()+tn.sum()+fp.sum()+fn.sum()))

        return tp, tn, fp, fn

if __name__ == "__main__":
    # Initialise metrics object
    m = metrics()

    # Set input data
    y_true = [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 5]
    y_pred = [1, 1, 1, 2, 2, 3, 4, 4, 1, 4, 4]

    # Compute metrics
    print(m.compute(y_true, y_pred))

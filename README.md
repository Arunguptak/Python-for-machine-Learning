What Is Machine Learning:- 
Machine Learning is the science (and art) of programming computers so they can learn from data. 
[Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed. 
A computer program is said to learn from experience E with respect to some task T and some performance measure P,
if its performance on T, as measured by P, improves with experience E. 

Why Use Machine Learning:-

1. First you would look at what spam typically looks like. You might notice that some words or 
phrases (such as “4U,” “credit card,” “free,” and “amazing”) tend to come up a lot in the subject.
Perhaps you would also notice a few other patterns in the sender’s name, the email’s body, and so on.
 You would write a detection algorithm for each of the patterns that you noticed, 
 and your program would flag emails as spam if a number of these patterns are detected. 
 3. You would test your program, and repeat steps 1 and 2 until it is good enough.
 Types of Machine Learning Systems:-
 
 Whether or not they are trained with human supervision (supervised,
 unsupervised, semisupervised, and Reinforcement Learning) • Whether or not they can learn incrementally on the fly (online versus batch learning) • Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model,
 much like scientists do (instance-based versus model-based learning) 
 The ROC Curve :-
 The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. It is very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) against the false positive rate. The FPR is the ratio of negative instances that are incorrectly classified as positive. It is equal to one minus the true negative rate, which is the ratio of negative instances that are correctly classified as negative. The TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus 1 – specificity. To plot the ROC curve, you first need to compute the TPR and FPR for various threshold values, using the roc_curve() function:
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores) Then you can plot the FPR against the TPR using Matplotlib. This code produces the plot in Figure 3-6:
def plot_roc_curve(fpr, tpr, label=None):  
plt.plot(fpr, tpr, linewidth=2, label=label)   
plt.plot([0, 1], [0, 1], 'k--')   
plt.axis([0, 1, 0, 1])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()
Multiclass:- Classification Whereas binary classifiers distinguish between two classes, multiclass classifiers (also called multinomial classifiers) can distinguish between more than two classes.
Some algorithms (such as Random Forest classifiers or naive Bayes classifiers) are capable of handling multiple classes directly.
 Others (such as Support Vector Machine classifiers or Linear classifiers) are strictly binary classifiers
 
 Multilabel Classification:-
 Until now each instance has always been assigned to just one class. In some cases you may want your classifier to output multiple classes for each instance. For example, consider a face-recognition classifier: what should it do if it recognizes several people on the same picture? Of course it should attach one label per person it recognizes. Say the classifier has been trained to recognize three faces, Alice, Bob, and Charlie; then when it is shown a picture of Alice and Charlie, it should output [1, 0, 1] (meaning “Alice yes, Bob no, Charlie yes”). Such a classification system that outputs multiple binary labels is called a multilabel classification system.
Multioutput Classification:-
The last type of classification task we are going to discuss here is called multioutputmulticlass classification (or simply multioutput classification). It is simply a generalization of multilabel classification where each label can be multiclass (i.e., it can have more than two possible values). 
 Polynomial Regression:- a more complex model that can fit nonlinear datasets. Since this model has more parameters than Linear Regression, it is more prone to overfitting the training data, so we will look at how to detect whether or not this is the case, using learning curves, and then we will look at several regularization techniques that can reduce the risk of overfitting the training set. 

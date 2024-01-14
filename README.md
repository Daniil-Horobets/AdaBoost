# AdaBoost ("Adaptive Boosting"<sup> 1</sup>)

## Description

The goal of the algorithm is to find a final hypothesis with low error relative to a given distribution _D_ over the
training examples.  <br/>
&emsp;<small>_from the original paper By Yoav Freund, Robert E Schapire_<sup> 1</sup></small>

## Boosting

- **Analogy:**
    - Consult several doctors, based on a combination of weighted diagnoses – weight assigned based on the previous
      diagnosis accuracy.
- **How boosting works:**
    - Weights are assigned to each training tuple.
    - A series of _k_ classifiers is iteratively learned.
    - After a classifier _M<sub>i</sub>_ is learned, the weights are updated to allow the subsequent classifier,
      _M<sub>i+1</sub>_ to pay more attention to the training tuples that were misclassified by _M<sub>i</sub>_.
    - The final _M<sup>*</sup>_ combines the votes of each individual classifier, where the weight of each classifier’s
      vote is a function of its accuracy.
- **Classification:**
    - Each classifier _M<sub>i</sub>_ returns its class prediction.
    - The bagged classifier _M<sup>*</sup>_ counts the votes and assigns the class with the most votes to X.
- **Boosting algorithm can be extended for numeric prediction.**

## AdaBoost.M1 Algorithm
&emsp;<small>_from the original paper By Yoav Freund, Robert E Schapire_<sup> 1</sup></small><br/><br/>

&emsp;_reload page if the color scheme doesn't match your theme_

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/Algorithm-dark.png" width="615" height="695">
  <img alt="Algorithm-light img" src="assets/Algorithm-light.png" width="615" height="695">
</picture>

## Simplified Step by Step Interpretation

### Training:

- Given a data set _D_ of _d_ class-labeled tuples: (x<sub>1</sub>, y<sub>1</sub>), ... ,(x<sub>d</sub>, y<sub>
  d</sub>)   
  with y<sub>d</sub> ∈ Y = {1, ... ,c}.
- Initialize empty lists to hold information per classifier: **w**, **β**, **M** ← empty list.
- Initialize weights for first classifier to hold same probability for each tuple: w<sub>j</sub><sup>1</sup> ← $\LARGE
  \frac{1}{d}$
- Generate _K_ classifiers in _K_ iterations. At iteration k,
    1. Calculate “normalized” weights:
       <div style="text-align: center;">$\LARGE \textbf{p}^k = \frac{\textbf{w}^k}{Σ_{j=1}^d w_j^i}$</div>
    2. Sample dataset with replacement according to **p**<sup>k</sup> to form training set _D<sub>k</sub>_.
    3. Derive classification model _M<sub>k</sub>_ from _D<sub>k</sub>_.
    4. Calculate error _ε<sub>k</sub>_ by using _D<sub>k</sub>_ as a test set as follows:
       <div style="text-align: center;">$\LARGE ε_k = Σ_{j=1}^d p_j^k \cdot \text{err}(M_k, x_j, y_j)$,</div> 
       where the misclassification error $\text{err}(M_k, x_j, y_j)$ returns 1 if M<sub>k</sub>(x<sub>j</sub>) $\neq$ y<sub>j</sub>, otherwise it returns 0.  
    5. If $\text{error}(M_k)$ > 0.5: Abandon this classifier and go back to step 1.
    6. Calculate
       <div style="text-align: center;">$\LARGE \textbf{β}_k = \frac{ε_k}{1 - ε_k}$.</div>
    7. Update weights for the next iteration:
       <div style="text-align: center;">$\LARGE  w_j^{k+1} = w_j^kβ_k^{1−\text{err}(M_k, x_j, y_j)}$.</div>
       If a tuple is misclassified, its weight remains the same, otherwise it is decreased. Misclassified tuple weights are increased  relatively.  
    8. Add **w**<sup>k+1</sup> , _M<sub>k</sub>_ , and _β<sub>k</sub>_ to their respective lists.

_See the implementation of the Training part in the `fit` function. You can [view it in adaboost.py](adaboost.py)_

### Prediction:

- **Initialize weight of each class to zero.**
- **For each classifier** _i_ **in** _k_ **classifiers:**
    1. Calculate the weight of this classifier’s vote:
       <div style="text-align: center;">$\LARGE  w_i = \log (\frac{1}{β_i})$.</div>
    2. Get class prediction _c_ for (single) tuple _x_ from current weak classifier $M_i: \quad c = M_i(x)$.
    3. Add _w<sub>i</sub>_ to weight for class _c_.
- **Return predicted class with the largest weight.**
- Mathematically, this can be formulated as:
     <div style="text-align: center;">$\LARGE  M(x) = \text{argmax}_{y∈Y} Σ_{i=1}^k (\log (\frac{1}{β_i}))M_i(x)$.</div>

_See the implementation of the Prediction part in the `predict` function. You can [view it in adaboost.py](adaboost.py)_

## Dataset

    dataset = pd.read_csv("dataset/car_train.csv")
    dataset.head()

<br/>
<img src="assets/Dataset.png" alt="Image" width="391" height="146"> <br/>

This dataset is a slightly modified version of
the [car evaluation dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
from the UCI Machine Learning Repository. Originally, this dataset has four class values. For the sake of this example
dataset modified to binary classification.<br/>

## Output Metrics

| Metric                     | Value |
|----------------------------|-------|
| Accuracy                   | 0.89  |
| Recall                     | 0.75  |
| Specificity                | 0.94  |
| Area Under the Curve (AUC) | 0.85  |
| F1 score                   | 0.80  |

<br/>

Confusion Matrix: <br/>

<img src="assets/confusion_matrix.png" alt="Image" width="384" height="288"> <br/>

Receiver Operating Characteristic (ROC) curve: <br/>

<img src="assets/roc_curve.png" alt="Image" width="384" height="288"> <br/>

_Values may differ in each run_ <br/>
_See implementation and more metrics calculation in [main.py](main.py)_

## Bibliography

### <sup>1 </sup>A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting<br/> &emsp;<small>_Under an Elsevier [user license](http://www.elsevier.com/open-access/userlicense/1.0/)_</small>

Yoav Freund, Robert E Schapire, <br/>
A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting, <br/>
Journal of Computer and System Sciences, <br/>
Volume 55, Issue 1, <br/>
1997, <br/>
Pages 119-139, <br/>
ISSN 0022-0000, <br/>
https://doi.org/10.1006/jcss.1997.1504. <br/>
(https://www.sciencedirect.com/science/article/pii/S002200009791504X)<br/>

### References

Code and Theory constitutes a component of the "Knowledge Discovery in Databases" course [exercise](https://github.com/FAU-CS6/KDD/tree/main/exercise/4-Classification-AdaBoost) offered by
Friedrich-Alexander-Universität Erlangen-Nürnberg <br/>
&emsp;<small>_Under GNU General Public License v3.0_</small>

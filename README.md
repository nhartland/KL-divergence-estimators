# KL divergence estimators

Here I test a few implementations of a KL-divergence estimator
based on k-Nearest-Neighbours probability density estimation.

The estimator is that of 

> Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.

# Estimator implementations


 - *naive_estimator*

    KL-Divergence estimator using brute-force (numpy) k-NN

 - *scipy_estimator*

    KL-Divergence estimator using scipy's KDTree

 - *skl_estimator*

    KL-Divergence estimator using scikit-learn's NearestNeighbours


# Tests



## 1-D self-divergence
 Estimate the divergence between two samples of size `N` and dimension
    1, drawn from the same ~ N(0,1) probability distribution.

|    Estimator    |   D(P|Q) | Time (s)|
|-----------------|----------|---------|
|naive_estimator  | 5.367e-03|0.0571733|
|scipy_estimator  | 5.367e-03|0.1456988|
|skl_estimator    | 5.367e-03|0.4575222|



## 2-D self-divergence
 Estimate the divergence between two samples of size `N` drawn
    from the same 2D distribution with
    mean=[0,0] and covariance=[[1, 0.1], [0.1, 1]] 

|    Estimator    |   D(P|Q) | Time (s)|
|-----------------|----------|---------|
|naive_estimator  |-5.814e-02|0.0623012|
|scipy_estimator  |-5.814e-02|0.1501002|
|skl_estimator    |-5.814e-02|0.4617238|



## 1-D divergence of Gaussians
 Estimate the divergence between two samples of size `N` and dimension
    1. The first drawn from N(0,1), the second from N(3,1).

|    Estimator    |   D(P|Q) | Time (s)|
|-----------------|----------|---------|
|naive_estimator  | 1.173e+00|0.0578928|
|scipy_estimator  | 1.173e+00|0.1400411|
|skl_estimator    | 1.173e+00|0.4653130|




## Requirements

- Python >= 3.6
- scipy, scikit-learn 
- slaypni/universal-divergence
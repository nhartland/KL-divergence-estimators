# KL divergence estimators

Here I test a few implementations of a KL-divergence estimator
based on k-Nearest-Neighbours probability density estimation.

The estimator is that of 

> Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.

# Estimator implementations


 - **naive_estimator**

    KL-Divergence estimator using brute-force (numpy) k-NN

 - **scipy_estimator**

    KL-Divergence estimator using scipy's KDTree

 - **skl_estimator**

    KL-Divergence estimator using scikit-learn's NearestNeighbours


# Tests



## 1-D self-divergence
 Estimate the divergence between two samples of size `N` and dimension
    1, drawn from the same ~ N(0,1) probability distribution.

|    Estimator    |  D(P\|Q) | Time (s)|
|-----------------|----------|---------|
|naive_estimator  | 2.032e-02|1.2554321|
|scipy_estimator  | 2.032e-02|2.4017279|
|skl_estimator    | 2.032e-02|3.5626271|

#### Convergence of estimator with *N*
![Convergence Plot](ConvergencePlots[test.filename])



## 2-D self-divergence
 Estimate the divergence between two samples of size `N` drawn
    from the same 2D distribution with
    mean=[0,0] and covariance=[[1, 0.1], [0.1, 1]] 

|    Estimator    |  D(P\|Q) | Time (s)|
|-----------------|----------|---------|
|naive_estimator  |-1.156e-04|1.8913550|
|scipy_estimator  |-1.156e-04|4.1763389|
|skl_estimator    |-1.156e-04|3.6734109|

#### Convergence of estimator with *N*
![Convergence Plot](ConvergencePlots[test.filename])



## 1-D divergence of Gaussians
 Estimate the divergence between two samples of size `N` and dimension
    1. The first drawn from N(0,1), the second from N(3,1).

|    Estimator    |  D(P\|Q) | Time (s)|
|-----------------|----------|---------|
|naive_estimator  | 3.455e+00|1.3573787|
|scipy_estimator  | 3.455e+00|2.6382170|
|skl_estimator    | 3.455e+00|4.2270157|

#### Convergence of estimator with *N*
![Convergence Plot](ConvergencePlots[test.filename])




## Requirements

- Python >= 3.6
- scipy, scikit-learn 
- slaypni/universal-divergence
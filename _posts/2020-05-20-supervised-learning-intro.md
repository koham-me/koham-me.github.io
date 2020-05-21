---
layout: post
title: "Supervised Learning and Linear Models"
date: 2020-05-20
tags: [datascience, ml, ai]
comments: true

---


## Function
Function is mathematical object which represents relationship between two sets. We can also say it is rule which associates element of two sets. We say $$f : X \to Y $$, and call $$X$$ domain of function and $$Y$$ co-domain of function.  
![Sets]({{site.baseurl}}/img/function-sets.svg)


For example,  
$$ f: R \to R , f(x) = x^2 $$  
In this case $$X$$ and $$Y$$ both are set or Real Numbers $$R$$. Which can be plotted on x-y axis as below,
![Graph of $$f(x)=x^2$$]({{site.baseurl}}/img/xsqaure.png)

Example of some well knows functions,

$$ E = mc^2 $$ , Famous energy mass equivalence relation by Einstien.

$$ F = ma $$, Netwons second law of motion.

$$ F = G\frac{m_1 m_2}{r^2} $$ Newtons law for gravitational force.

These functions are important, they help in making decisions, e.g. I can determine amount of force needed to accelerate a car with $$10ms^{-2}$$ based on its mass, which can help me with the amount of petrol needed for that, or how much of the uranium is sufficient to generate electricty for a city.

### Function Approximation 
If we have rule or function structure, we can compute y(dependent variable) for any value of x(independent variable). But in some cases we do not know the exact nature of function, all we can do is observe values of x and y for some finite number of cases, for example,

| $$x$$      |  $$y$$   |
|-------:|----------:|
| 0      |  0        |
| 1      |  1        |
| 2      |  4        |
| 3      |  9        |
| 4      |  16        |
| 5      |  25        |

From these finite number of observations, we want to find true nature of function, so that we can use it to compute or predict y(dependent variable) for any given value of x(independent variable). Since we have only limited number of observation, we may not be able to get the true nature of the function, but we can still try to get an approximation of it. This method is called **function approximation**. In other word function approximation is to search for the approximate function for the true function in the space of all functions, this has to be supplied with error measure to find how much it deviates from the true nature of function based on the observations from the true function, we want to have as close as possible with respect to the error measure.


### Modelling

As we have seen in previous section, function approximation is method to search for the a function close to true function in space of all functions. The space of all functions will be large and it will have infinitely many functions, how do we restrict ourselves to some subset of all functions, such that the search space is smaller and our search procedure becomes feasible. We can do this with some assumption about the true nature of the function and then based on observation we can decide the best approximation of true function, this is called as modelling or model building, since we do not know the true. If we assume below rectangle as set of all possible functions, then single point is one particular function.
![Space of all Functions]({{site.baseurl}}/img/function_space.png)

We may consider function with some parameters lets say $$\theta$$ then we can take, $$f\hat(x;\theta)$$ as approximation of $$f(x)$$ and then try to find $$\theta$$, based on the observations, for example

$$ f\hat(x;\theta) = x + x^2 + x^3 + ... + x^\theta $$  

$$ f\hat(x;\theta_0, \theta_1) = \theta_0 + \theta_1 x $$  

$$ f\hat(x;\theta) = e^{-\theta^Tx} $$ where  $$x,\theta \in R^n $$  

Consider following set of observations generated from true function $$f(x) = 3.6 + 1.2 x^{1.001} $$,

|x    | y    |
|----:|--:   |
|1.0  | 4.8  |
|1.5  | 5.4  |
|1.6  | 5.5  |
|2.0  | 6.0  |
|2.2  | 6.24 |
|2.7  | 6.84 |
|3.6  | 7.92 |
|3.9  | 8.2  |

If we draw a scatter diagram of above set of observations we get following, and it is reasonable to assume linear relationship here an

| | |
|--:|:--|
| $$ f\hat(x; a, b) = a + bx $$ | ![Scatter Diagram]({{site.baseurl}}/img/scatter_1001.png)|

## Supervised Learning
Till now we have assumed that the relationship is deterministic, i.e. for a given value of x, y will be same all the time, but this might not be true in every case. There might be some randomness, and for same value of x, we might observe different value for y. This randomness may be due to error in the process of measurement , or it might be inherent in the process generating these observations. In the presence of randomness, our modelling process changes to accomodate the randomness in the modelling, we assume that the true nature of the relationship between x and y as below

$$ Y = f(x) + \epsilon $$  
and where $\epsilon$ is uncorrelated random variable(error) statistically independent of $$x$$.

$$ E[\epsilon] = 0  \\ Corr(\epsilon_i, \epsilon_j)=0$$

We would like to estimate or approximate a function $f$ with some parametric function $f\hat(x;\theta)$, so that we can predict 

$$\hat Y = f\hat(x;\theta)$$ 
for any value of x.

This function approximation(parameter estimation) is called **supervised learning** for given pair of finite observations $$O = (x_1, y_1),(x_2, y_2), .... (x_n, y_n)$$. When y is limited to take only finite set of values it is called **classification** and when value of y is continuous it is called as **regression**.  We call set of all observtions available to us as training set, hence supervised learning is method to approximate or estimate the function in presence of randomness with the available training data.


### Measuring Quality of Estimated Function

In order to to evalute the performance of estimated function, we need to measure, how well the estimated values match with observed values. Thus we want to quantify the extent of closeness between estimated and observed values. One of the measure is *mean square error* 

$$ MSE = \frac{1}{n} \sum_{i=0}^n (y_i - \hat f(x_i))^2 $$

MSE computed from the training data is *training MSE*, but our purpose is to compute error in estimation with respect to unkown values for x such that we can generalize the estimated function for all values of x. One of simplest way to do is to devide set of available data into training and test data, use test data for computing test MSE, which are unseen in training process. There are more robust method to this, which are not discussed here.

In classification setting we can compute error rate with

$$ Error = \frac{1}{n} \sum_{i=0}^n (y_i \ne \hat f(x_i)) $$


## Simple Linear Regression
![Simple Linear Regression]({{site.baseurl}}/img/slm.png)

We assume that the nature of true function is linear

$$ f = \beta_0 + \beta_1 x $$

and we would like to estimate this function based on the training observations with function, which boils down to estimating values of parameters $$\beta_0, \beta_1$$, lets say our estimate for these are $$\hat \beta_0, \hat \beta_1$$, then we can write our approximated or estimated function as 

$$ \hat f = \hat \beta_0 + \hat \beta_1 x $$

Interpretation of $$\beta_0$$ and $$\beta_1 $$
![Interpretation of Parameters]({{site.baseurl}}/img/inter.png)

### Parameter Estimation
$\beta_0$ and $\beta_1$ are parameters to be estimated from the available training data, we can do this by minimizing training RSS, which measures deviation from the observed and estimated values as follows, 

$$ RSS = \sum_{i=0}^n (y_i - \hat f(x_i))^2 \\
       = e_1^2 + e_2^2 + .. + e_n^2 $$
       
where $$e_1, e_2, .., e_n $$ are called residuals.
       

We want to minimize RSS with respect to the parameters we want to estimate and hence we can obain parameters by setting the first derivative of the training MSE as 0, and compute the parameters

$$\frac{d}{d\hat \beta_0} RSS = -2 \sum_{i=0}^{n}(y_i - \hat \beta_0 - \hat \beta_1 x_i)  = 0$$

$$\frac{d}{d\hat \beta_1} RSS = -2 \sum_{i=0}^{n}(y_i - \hat \beta_0 - \hat \beta_1 x_i)x_i = 0$$

By solving these equations we obtaing optimal values as below

$$\hat \beta_0 =  \bar y - \beta_1 \bar x$$ 

$$\hat \beta_1 =  \frac{\sum_{i=0}^n (x_i - \bar x) (y_i - \bar y)}{\sum_{i=0}^n (x_i - \bar x)^2}$$ 


### Model Assesment

#### Residual Standard Error
Due to the presence of these error terms, even if we knew the true regression line (i.e. even if $$\beta_0$$ and $$\beta_1$$ were known), we would not be able to perfectly predict Y from X. The RSE is an estimate of the standard deviation of $$\epsilon$$ Roughly speaking, it is the average amount that the response will deviate from the true regression line.

$$ RSE = \sqrt \frac{\sum_{i=0}^n (\hat y_i - y_i)^2}{n-2} \\
       = \sqrt \frac{RSS}{n-2}$$
       
The RSE is considered a measure of the lack of fit of the model to the data. If the predictions obtained using the model are very close to the true outcome values then RSE will be small, and we can conclude that the model fits the data very well. On  the other hand, if $\hat y_i$  is very far from $y_i$ for one or more observations, then the RSE may be quite large, indicating that the model doesn’t fit the data well. 

#### $$ R^2 $$ = Coefficient of Determination.

It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.

$$ R^2 = \frac{Regression Variability}{Total Variability} \\
       = \frac{TSS - RSS}{TSS}
$$ 

where is $$ TSS = \sum_{i=0}^n (y_i - \bar y)^2 $$ total sum of squares. 

TSS measures the total variance in the response Y, and can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, RSS measures the amount of variability that is left unexplained after performing the regression. Hence, TSS − RSS measures the amount of variability in the response that is explained (or removed) by performing the regression, and $$R^2$$ measures the proportion of variability in Y that can be explained using X.

This implies that $$R^2%$$ of the variability of the dependent variable has been accounted for, and the remaining $$(1-R^2)%$$ of the variability is still unaccounted for. Hence more the value of $$R^2$$ better is fit of the model.


Recall that we have made following assumptions
1. Random errors are independent of X.
2. Random errors are uncorrelated.
3. Random errors have common variance.

These assumptions will have to be true for the residuls our model produces. If we see any deviation from these assumptions, we can say our model is not proper. We can do inspection for these with the help of graph

![Residual Analysis]({{site.baseurl}}/img/residual2.png)
![Residual Analysis]({{site.baseurl}}/img/residual3.png)

## Multiple Linear Regression

When we have more than one variable, which means our function is of the form $$f:R^p \to R$$, we have to consider all p variables when modelling.

$$ y = f(x) + \epsilon, x \in R^p $$

One way of solving this probelm is to consider all variables separaltely in simple linear regression and asses the relationship, but we can not combine the results from different simple linear regression to get the result. Second thing when we consider modelling one variable at a time, we are leaving behind p-1 variables combined influence at it. 

Better approach is to consoder linear model with all p variable, in below form

$$ f(x_1, x_2, .. x_p) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p $$

and we would like to approximate $y$ with

$$ \hat y = \hat f(x_1, x_2, .. x_p) = \hat\beta_0 + \hat\beta_1 x_1 + \hat\beta_2 x_2 + ... + \hat\beta_p x_p $$

If we assume $$x_0 =1$$ for all observations we can rewrite above as

$$ \hat y = \sum_{i=0}^p \hat \beta_i x_i = \beta^T x \\
    \hat Y = \hat\beta X $$

where $$\hat Y$$ is vector of all dependent variable and $$X$$ is $$n \times p$$ matrix consisting all independent variables. 

### Parameter Estimation

$$ RSS = \sum_{i=0}^n (y_i - \hat\beta_0 - \hat\beta_1 x_1 - \hat\beta_2 x_2 - .. - \hat\beta_p x_p) ^ 2 $$  

We can write this in matrix form as below

$$ RSS(\hat\beta) = (Y - \hat\beta X)^T( Y - \hat\beta X) $$

For optimal values of $\beta$, the derivative of above will be 0. which means

$$ \frac{d}{d\hat\beta} RSS = 0 \\
   -2X^T(Y - X\hat\beta) = 0 \\
   \hat\beta = (X^T X)^{-1} X^T Y $$
   
### Model Assesment
1. Residual Standard Error

$$ RSE = \sqrt \frac{\sum_{i=0}^n (\hat y_i - y_i)^2}{n-p-1} \\
       = \sqrt \frac{RSS}{n-p-1}$$

2. $$R^2=$$ Coefficient of Determination

3. Adjusted $$R^2=$$
In general, $$R^2$$ never decreases when a regressor is added to the model, regardless of the value of the contribution of that variable. The adjusted $$R^2$$ penalizes us for adding terms that are not helpful.

$$ \hat R^2 = 1 - \frac{\frac{RSS}{n-p}}{TSS} $$

4. Graphical assesment for the assumptions similar to simple linear regression.


## Logistic Regression
### Linear regression approach
For two class(when y can take only two values), we can put 0 for one and 1 for other class and use linear regression approach and can then put a additional rule as,

$$y_i = 1 , when \hat f(x_i) \ge 0.5 $$

$$y_i = 0, otherwise $$  

which means we can think of $$\hat f(x) $$ as the probability of x being from class 1.

This might be a good idea, but some of the time linear regression can produce values outside range of $$[0,1]$$, which will be harder to interpret in terms of probability. Second thing we can not extend it for more than two classes. 

### Logistic Model
To avoid probelm of linear regression values outside range of $$[0,1]$$, we can use logistic function, which is of the form

$$ f(a) = \frac{e^a}{1+e^a} $$  

this function always takes values in interval $$[0,1]$$ and hence we can use it with linear regression, assuming it as probability for one of the classes

$$p(x) = \frac{e^{\beta X}}{1 + e^{\beta X}} $$

In other way we can write 

$$  \frac{p(x)}{1-p(x)} = e^{\beta X} \\
    log\frac{p(x)}{1-p(x)} = \beta X $$

This quantity is log odd of being from class 1, which is also called logit. Which can be then approximated with linear regression aproach. 

### Parameter Estimation
We can utilise principle of maximum likelihood to estimate the parameters of logistic regression. 

$$ l(\hat \beta) = \Pi_{i:y_i=1}p(x_i) \Pi_{j:y_j=0}(1-p(x_j)) $$

$$\hat \beta$$ parameters are chosen, such that likelihood is maximized. This can be achieved with gradient descent like optimzation, and optimal values of parameters can then be utilized for prediction.

### Model Assesment

#### Error Rate 

$$ Error Rate= \frac{1}{n} \sum_{i=0}^n I(y_i \ne \hat f(x_i)) $$

where I is indicator function.

#### Accuracy
This represents number of observations classified correctly by our model. 

$$ Accuracy = \frac{1}{n} \sum_{i=0}^n I(y_i == \hat f(x_i)) $$

Accuracy is nothing but 1 - Error Rate.

#### Confusion Matrix
![Confusion Matrix]({{site.baseurl}}/img/confusion1.png)

#### ROC - Reciever Operating Characteristics
In our case we assumed a threshold of 0.5 to decide the classes, however it may be the case that one class is more probable than the other and hence threshold may change. The overall performance of a classifier, summarized over all possible thresholds, is given by the area under the (ROC) curve (AUC). An ideal ROC curve will hug the top left corner, so the larger the AUC the better the classifier.
![ROC]({{site.baseurl}}/img/roc.png)

## Regularization

### Bias-Variance Tradeoff

The residual sum error in the model can be decomposed in three components,
1. Irreducible error, becuase of presence of random component.
2. Bias , due to modelling particular form of the true function.
3. Variance, due to closely following random noises in approximation.

As the complexity/flexibility of the model increase, the bias decreases, however variance increases as it tries to model even random components. For the less flexible models, bias is higher due to modelling assumption. Trying to decrease one, increaded the other, this is called as bias-variance tradeoff. Flexibility of the model increase variance term , in which case it might be able to predict all data in training set but fail to give good prediction for test set or generalize, this is called as overfitting. Method to increase biasness to reduce variance is termed as regularization.
![Bias-Variance Tradeoff]({{site.baseurl}}/img/bias-var.png)

In case of linear models, more number of variables we include in our model, more flexible it becomes in modelling the true nature of the function. With the increase in flexibility biasness of the model deceases, however variance starts increasing. Which results in overfitting. We can avoid overfitting with some of the following approaches.


### Best Subset Selection
To perform best subset selection, we fit a separate least squares regression for each possible combination of the p predictors. That is, we fit all p models that contain exactly one predictor, and then all models that contain exactky two predictors, and so forth. We then look at all of the resulting models, with the goal of identifying the one that is best with respect to measures such as RSE, $$R^2$$ or error rate etc. The problem of selecting the best model from among the $2^p$ possibile models is called best subset selection.

This is infeasible for even small value of p, e.g for p = 30, we have 1073741824 candidate models to select from.

## Shrinkage Methods
As an alternative, we can fit a model containing all p predictors using a technique that constrains or regularizes the coefficient estimates, or equivalently, that shrinks the coefficient estimates towards zero. Two of such methods are ridge regression and lasso for linear regression. 


### Ridge Regression

$$ RSS = \sum_{i=0}^n (y_i - \hat\beta_0 - \hat\beta_1 x_1 - \hat\beta_2 x_2 - .. - \hat\beta_p x_p) ^ 2 $$  

Instead of minimizing above to estimate parameters, we minimize a different function, which is

$$ RSS + \lambda \sum_{j=1}^p \beta_j^2 = \sum_{i=0}^n (y_i - \hat\beta_0 - \hat\beta_1 x_1 - \hat\beta_2 x_2 - .. - \hat\beta_p x_p) ^ 2  + \lambda \sum_{j=1}^p \beta_j^2$$

$$\lambda \ge 0 $$ is called as tuning parameter.

How is this helpful??

The second term is called a shrinkage penalty, and it is small when $$\beta_1, , . . . ,\beta_p$$ are close to zero, and so it has the effect of shrinking the estimates of $$\beta_j$$ towards zero. The tuning parameter $$\lambda$$ serves to control parameter. When $$\lambda = 0$$, the penalty term has no effect, and ridge regression will produce the least squares estimates. However, as $$\lambda \to \infty$$, the impact of the shrinkage penalty grows, and the ridge regression coefficient estimates will approach zero.

Ridge regression’s advantage over least squares is rooted in the bias-variance trade-off. As $\lambda$ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias.

### LASSO
Ridge regression does have one obvious disadvantage. Unlike best subset, which will generally selects models that involve just a subset of the variables, ridge regression will include all p predictors in the final model. This may not be a problem for prediction accuracy, but it can create a challenge in model interpretation in settings in which the number of variables p is quite large. The lasso is a alternative to ridge regression that overcomes this disadvantage.

In LASSO we try to minimize 

$$ RSS + \lambda \sum_{j=1}^p |\beta_j| = \sum_{i=0}^n (y_i - \hat\beta_0 - \hat\beta_1 x_1 - \hat\beta_2 x_2 - .. - \hat\beta_p x_p) ^ 2  + \lambda \sum_{j=1}^p |\beta_j| $$

As with ridge regression, the lasso shrinks the coefficient estimates towards zero. However, in the case of the lasso, the $$L1$$ penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter $$\lambda$$ is sufficiently large. Hence, much like best subset selection, the lasso performs _variable selection_. As a result, models generated from the lasso are generally much easier to interpret than those produced by ridge regression.

I hope this article was helpful to understand some of the aspects of linear models, for any suggestions or queries, please put your comments.

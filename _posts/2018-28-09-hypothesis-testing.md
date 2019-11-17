---
layout: post
title: "Hypothesis Testing"
date: 2018-09-28
tags: [statistics, machinelearning]
comments: true

---

A statistical hypothesis is conjecture about the population parameter which may or may not be true. A simple example of conjecture would be average height of all children of age 14 years is 135cm. We need observations to verify the validity of the hypothesis. The verification procedure from observed data is called hypothesis testing. Lets consider another example, we want to verify that the acceleration due to gravity on earth and on moon is same or not. When we do hypothesis testing we have to split hypothesis into two parts, NULL hypothesis and alternate hypothesis. Based on the observed data we decide whether we would reject NULL hypothesis or do not reject null hypothesis. Generally we make null hypothesis which we want to disapprove, such that rejection of it, approves the alternate hypothesis. Failing to reject null hypothesis does not mean that we are accepting null hypothesis, it means there are not enough evidence in data to reject it. For example in case of jury, following are null and alternate hypothesis,

*H0(Null Hypothesis) : Defendant is innocent.*  
*H1(Alternate Hypothesis) : Defendent is guilty.*

In our example of acceleration due to gravity, we would have  
*H0 : Acceleration due to gravity on moon is same as on earth. ($$ g_e = g_m $$)*  
*H1 : Acceleration due to gravity on moon is not same as it is on earth. ($$ g_e \neq g_m $$)*

In case of height of children,  
*H0 : Average height of children of age is 135cm.*  
*H1 : Average height of children of age 14 is not 135cm.*

Based on evidences jury decides to reject or failure to reject null hypothesis, which means either defendant is guilty or there is not enough evidence that he is guilty. In such decision there are two ways to commit mistakes and these are,

| | Defendant is innocent | Defendant is guilty
Freed | Correct Decision | Wrong Judgement
Convicted | Wrong Judgement | Correct Decision

In statistics these are called type I and type II errors

| | H0 is true | H1 is true
Do not reject H0| Correct Decison | Type II Error
Reject H0| Type I Error | Correct Decision

Acceptance of H1 when H0 is true is called type I error. Probability of committing type I error is called level of significance. Maximum probability type I error permitted is called critical value. In case of jury example would be convicting the defendant when he is innocent.
Below are step by step procedure to perform hypothesis testing,
* State the hypothesis(Null hypothesis as well as alternate hypothesis).
* Find out the critical value of the experiment.
* Define the test statistics and compute the test value.
* Based on comparison of test value and critical value makde decision to reject or not to reject null hypothesis.


### z-Test for Mean

We will continue with the height example, We are going to test the hypothesis based on below data. Please note this is simulated data.

|Height in cm |
|123.67|
|116.50|
|105.53|
|124.58|
|118.05|
|134.96|
|..|
|120.89|
|131.76|
|126.23|
|132.97|

Step 1: State the hypothesis  
*H0 : Average height of children of age is 135cm.*  
*H1 : Average height of children of age 14 is not 135cm.*

Step 2: Let us assume that we are going to ignore 10% type I error and hence critical value will be 0.10.  
Step 3: We need to define a test statistics, in case we know true variance we can define test statistics as

$$ z = \frac{\bar{X} - \mu} {\frac{\sigma}{\sqrt{n}}} $$

Above is called z-statistics and follws standard normal distribution under the assumptions of null hypothesis. If it deviates from standard normal distribution more than critical value, we can say that the assumption of null hypothesis is not proper and hence reject null hypothesis. In our case we got value for z-statistics as -3.30 and z-statistics corresponding to critical value is 1.64. Which means we can accept any value in between (-1.64, 1.64). But -3.30 is out of this range.

```python
#sample code to simulate above data and calculate z-statistics
import numpy as np
dataset = np.random.normal(loc=132, scale=10, size = 100)
z = np.sqrt(100) * (dataset.mean() - 135) / 10 
```

Step 4: We reject the null hypothesis, that the average height of 14 year children is 135cm.

In case we do not know the variance value beforehand, we will get the t-statistics which can be computed as 
$$ t = \frac{\bar{X} - \mu}{\frac{s.d.}{\sqrt{n}}} $$ 

s.d. is standard deviation calculated from data. The correspnding test is called t-test.

#### p-value
The P-Value (or probability value) is the probability of getting a sample statistic (such as the mean) or a more extreme sample statistic in the direction of the alternative hypothesis when the null hypothesis is true.

Instead of comparing with z or t-statistics, we can compute the corresponding p-value and reject the null hypothesis if p-value found is smaller than the critical value. In above case p-value would be 0.0010 which is much smaller than 0.10, hence we can safely reject the null hypothesis.

### Two Sample Test

Continuing with example of comparison of acceleration due to gravity on earth surface and moon surface. We know that a free falling ball from heigh h will take $$\sqrt{\frac{2h}{g}}$$ time to reach the surface. Consider the hypothetical experiment of dropping balls from height h on earth and on moon. We record the time it touches the ground. From this experiment we compute the value of g on earth and on moon. 


---
layout: post
title: "Sequential Minimal Optimization for Support Vector Machines"
date: 2020-06-14 13:32:20 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img:  # Add image post (optional)
---


#### In this blog, I will discuss about the Sequential Minimal Optimization(SMO) which is an Optimization technique that is used for training Kernel SVM's

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153085/SMO/0_JeeVKBFsgeg5Lpef_mnt0fk.jpg" /></div>

<div style="text-align:center"><p>Photo by <a href="https://unsplash.com/@ikukevk?utm_source=medium&amp;utm_medium=referral">Kevin Ku</a> on <a href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral">Unsplash</a></p>
</div>

In this blog, I will discuss the Sequential Minimal Optimization(SMO) which is an Optimization technique that is used for training Support Vector Machines(SVM). Before getting into the discussion on SMO, I will discuss some basic Mathematics required to understand the Algorithm. Here I will start with the discussion on Lagrange Duality, Dual Form of SVM and then solving the dual form using the SMO algorithm.

### Lagrange Duality

Consider the following Primal Optimization Problem :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153201/SMO/1_FcVnO7yOhpRX9t02o6edRw_lytwhd.gif" /></div>

To solve this, let us define the Lagrangian as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153245/SMO/1_8Ob5SdiN94XbREkEg_TkSQ_vkqf4o.gif" /></div>


Here the **_αi_**’s and **_βi_**’s are the Lagrange Multipliers. Now let us define the primal Lagrangian function as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153298/SMO/1_T0HFYpZvWW3-MBY1c9vifA_ysgskc.gif" /></div>


For any given **_w_**, if **_w_** violates any of the Primal constraints, then it can be shown that :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153329/SMO/1_2M6gKZFiRkHMkwRuPzha-Q_qfrtcn.gif" /></div>


Now if the constraints are satisfied for a particular value of **_w_**, then **_p(w)=f(w)_**. This can be summarised as follows :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153386/SMO/1_l8rat0Q9UGFjY5PgX_hBRg_htyur7.gif" /></div>


Now let’s consider the minimization problem :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153425/SMO/1_Vz39OrZMWUErjIFMj65VBQ_r7cthy.gif" /></div>

We have already seen that **_p(w)_** takes the same value as the objective for all values of **_w_** that satisfies the primal constraints and **_∞_** otherwise. So we can clearly see from the minimization problem that the solutions to this problem is same as the original Primal Problem. Now let us define the optimal value of the objective to be :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153461/SMO/1_x3DVlINk6qrd1st7ywrICw_zbsnoz.gif" /></div>


Now let us define the Dual Lagrangian function as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153503/SMO/1_IT9FlhaZZvRb20_9nXFkWg_vvjfnb.gif" /></div>


Now we can pose the Dual Optimization problem as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153542/SMO/1_gjBtN8-J1hWROtxDS_R2Rw_nxvsmu.gif" /></div>


Now let’s define the optimal value of the dual problem objective as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153593/SMO/1_Z8PgEcVdcvkKHOFdSDHoYQ_gubhxw.gif" /></div>


Now it can be shown that :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153633/SMO/1_mCGZVgSaO5lrFw9KXhUAbA_vwfca2.gif" /></div>


“Max-min” of a function will always be less than or equal to “Min-max”. Also under certain conditions, **_d∗=p∗_**. These conditions are :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153667/SMO/1_IAkKc4LuDIpq8P5BwHnKDg_fil22u.png" /></div>


This means that we can solve the dual problem instead of the primal problem. So there must exist **_w∗_**, **_α∗_**, **_β∗_** so that **_w∗_** is the solution to the primal and **_α∗_** and **_β∗_** are the solutions to the Dual Problem and **_p∗=d∗_**. Here **_w∗_**, **_α∗_**, **_β∗_** satisfy the Karush-Kuhn-Tucker(KKT) constraints which are as follows :

**Stationarity conditions :**

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595153696/SMO/1_7_DEo1Dqmnitj6_vYEz87g_ic5sez.png" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595154545/SMO/1_L2te3X-qpcndyyg_5xU3KQ_hdgu4d.png" /></div>

**Complimentary slackness condition :**

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595154585/SMO/1_l6yC-v8i4NMOvX5r79NIgQ_cqakdo.png" /></div>

**Primal feasibility condition :**

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595155859/SMO/1_6qAPww1iz3MVccwRcnurzg_cs1jde.png" /></div>


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595155894/SMO/1_07R5NYu-f-7ta00z2EHjDg_w1fxe2.png" /></div>


**Dual feasibility condition :**

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595155932/SMO/1_LNITOL1Icwdg5TaURYwVKg_zt1hvr.png" /></div>


#### Optimal Margin Classifiers :

The Primal Optimization problem for an Optimal margin Classifier is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595155964/SMO/1_0oXPJ2hgU4d_QO_ELgD6Kw_vkytk3.png" /></div>

Now let us construct the Lagrangian for the Optimization problem as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156000/SMO/1_aVEtXCRlNLEMydwMg2Lf9g_dl5jb0.png" /></div>


To find the dual form of the problem, we need to first minimize **_L(w,b,α)_** wrt. **_b_** and **_w_**, this can be done by setting the derivatives of **_L_** wrt. **_w_** and **_b_** to zero. So we can write this as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156032/SMO/1_DAEU3zVtnWWCaH97yw879g_jvc3rk.png" /></div>


For derivatives of **_L_** wrt. **_b_**, we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156073/SMO/1_hHQoe64gL_Q_-ySkQ2Np1w_solcee.png" /></div>

Now we can write the Lagrangian as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156117/SMO/1_Ku4PLKisZyEEYtUVufdzVg_qkufyq.png" /></div>

But we have already seen that the last term of the above equation is zero. So rewriting the equation, so we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156164/SMO/1__xVo6bnReT3IVeRVczv7ug_bqndsu.png" /></div>


The above equation is obtained by minimizing **_L_** wrt. **_b_** and **_w_**. Now putting all this together with all the constraints, we obtain the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156209/SMO/1_Y3FQE_bgHtLFpwtfRNK7Uw_rco1hi.png" /></div>


Now once we have found the optimal **_αi’s_** we can find optimal **_w_** using the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156243/SMO/1_ZXXB_nNgmGBkxn0IdoRtoQ_lpdbtf.png" /></div>


Having found **_w∗_**, now we can find the intercept term (**_b_**) using the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156296/SMO/1_QpXvdab_0eXfXbUWibdWnw_q6ilzh.png" /></div>


Now once we have fit the model’s parameters to the training set, we can get the predictions from the model for a new datapoint **_x_** using :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156343/SMO/1_vTSdgGj7Wb1S3GL6uqlNeg_a7pmcn.png" /></div>


The model will output 1 when this is strictly greater than 0, we can also express this using the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156393/SMO/1_tp_-3GLo_EBi4SdydJBwiQ_o6l4af.png" /></div>


### Sequential Minimal Optimization

Sequential Minimal optimization (SMO) is an iterative algorithm for solving the Quadratic Programming(QP.) problem that arises during the training of Support Vector Machines(SVM). SMO is very fast and can quickly solve the SVM QP without using any QP optimization steps at all. Consider a binary classification with a dataset **_(x1,y1),…..,(xn,yn)_** where **_xi_** is the input vector and **_yi∈{−1,+1}_** which is a binary label corresponding to each **_xi_**. An optimal margin SVM is found by solving a QP. problem that is expressed in the dual form as follows :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595156468/SMO/1_orXru39zbu4JjbJbW9ZeqQ_vnzlv9.png" /></div>


Here **_C_** is SVM hyperparameter that controls the tradeoff between maximum margin and loss and **_K(xi,xj)_** is the **Kernel Function**. **_αi_** is **Lagrange Multipliers**. SMO is an iterative algorithm and in each step, it chooses two Lagrange Multipliers to jointly optimize and then finds the optimal values for these multipliers and updates the SVM to reflect the new optimal values. The main advantage of SMO lies in the fact that solving for two Lagrange multipliers can be done analytically even though there are more optimization sub-problems that are being solved, each of these sub-problems can be solved fast and hence the overall QP problem is solved quickly. The algorithm can be summarised using the following:


``` 
Repeat till convergence :  
{
	1. Select some pair αi and αj  
	2. Reoptimize W(α) wrt. αi and αj while holding all other αk’s fixed.
}
```
There are two components to SMO: An analytical solution for solving for two Lagrange Multipliers and a heuristic for choosing which multipliers to optimize.

#### Solving for two Lagrange Multipliers :

SMO first computes the constraints on the two multipliers and then solves for constrained minimum. Since there is a bounding constraint **_0≤αi≤C_** on the Lagrange multipliers, these Lagrange multipliers will lie within a box. The following causes the Lagrange multipliers to lie on the diagonal :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595160930/SMO/1_hiogcJw6aJ22MoVaZWu7WQ_eslylh.png" /></div>


If SMO optimized only one multiplier, it could not fulfill the above equality constraint at every step. Thus the Constrained minimum of the objective function must lie on a diagonal line segment. Therefore, one step of SMO must find an optimum of the objective function on a diagonal line segment. This can be seen from the following diagram :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595160976/SMO/1_BWWRI66as4Psu0_sNwzDNw_wizfad.png" /></div>

<div style="text-align:center"><p><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf">Source</a></p></div>



Because there are only two multipliers, the constraints can be easily displayed in two dimensions as seen above. The algorithm first computes the second Lagrange multiplier **_α2_** and then computes the ends of the diagonal line segment in terms of **_α2_**.

If **_y1≠y2_**, then the following bounds apply to **_α2_** :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161335/SMO/1_5rTGKL7G3ROEVWqKjTlbOQ_fwm0fv.png" /></div>


If **_y1=y2_**, then the following bounds apply to **_α2_** :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161381/SMO/1_QP-L3O-IT5dYXf3XX4Bd8w_rjkzih.png" /></div>


The second derivative of the Objective function along the diagonal is expressed as :

![](https://cdn-images-1.medium.com/max/800/1*XsehPyxFGZdHAokgZVWo0Q.png)

Under normal circumstances, **_η_** will be greater than zero and there will be a minimum along the direction of the linear equality constraint. In this case, SMO computes the minimum along the direction of the constraint :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161418/SMO/1_qa1rcFboK-Z9849ubHhsZw_somq7t.png" /></div>


Where :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161457/SMO/1_-CsJDbK5ETBlt0hDCX5ueg_gvaskh.png" /></div>


**_μi_** is the output of the SVM for the **_i_**’th training example. The constrained minimum is found by clipping the unconstrained minimum to the ends of the line segment :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161501/SMO/1_mPfGftqM0GUjYDf_Qo0qAQ_pfop2h.png" /></div>


The value of **_α1_** is computed from the new clipped **_α2_** using:

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161539/SMO/1_7z0Bpi4V_hw5obu92QZ94g_d4djwi.png" /></div>


Under normal circumstances **_η>0_**. A negative **_η_** occurs if the kernel **_K_** does not obey Mercer’s Conditions which states that for any Kernel(**_K_**), to be a valid kernel, then for any input vector **_x_**, the corresponding Kernel Matrix must be symmetric positive semi-definite. **_η=0_** when more than one training example has the same input vector **_x_**. SMO will work even when **_η_** is not positive, in which case the objective function(**_ψ_**) should be evaluated at each end of the line segment :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161578/SMO/1_U3HUCrYkNVHtcnfH7qZCuQ_ejl02h.png" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161616/SMO/1_hRJHHCWsEyrJGgSDvI3a5A_y5vwid.png" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161654/SMO/1_GCi2cVEe35CMgMsJAgpcaA_s5eeyu.png" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161699/SMO/1_8eNeQQRoRT-OHH4e7QwDcg_duxwpx.png" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161740/SMO/1_NcmDkXdWTHeudL_XjTZUWQ_w0jm7v.png" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161794/SMO/1_OIwbLbcb08Na6RYC0OWVyA_v489xb.png" /></div>


SMO will move the Lagrange multipliers to the point that has the lowest value of the Objective Function. If Objective function is the same at both ends and Kernel obeys Mercer’s conditions, then the joint minimization cannot make progress, in such situations we follow the Heuristic approach, which is given below.

#### Heuristic for choosing which multipliers to Optimize :

According to **Osuna’s Theorum**, a large QP. The problem can be broken down into a series of smaller QP sub-problems. A sequence of QP. subproblems that add at least one violator to the Karush-Kuhn-Tucker(KKT) conditions is guaranteed to converge. So SMO optimizes and alters two lagrange multipliers at each step and at least one of the Lagrange multipliers violates the KKT conditions before the next step, then each step will decrease the objective function and thus by the above theorem making sure that convergence does happen.

There are two separate heuristics: one for choosing the first Lagrange multiplier and one for the second. The choice for the first heuristic forms the outer loop of SMO. The outer loop iterates over the entire training set, and determines if each example violates the KKT conditions and if it does, then it is eligible for optimization.

After one pass through the entire dataset, the outer loop of SMO makes repeated passes over the non-bound examples(examples whose Lagrange Multipliers are neither 0 nor **_C_**) until all of the non-bound examples obey the KKT conditions within **_ϵ_**. Typically **_ϵ_** is chosen to be 10^(−3).

As SMO progresses, the examples that are at the bounds are likely to stay at the bounds while the examples that are not at the bounds will move as other examples are optimized. And then finally SMO will scan the entire data set to search for any bound examples that have become KKT violated due to optimizing the non-bound subset.

Once the first Lagrange multiplier is chosen, SMO chooses the second Lagrange multiplier to maximize the size of the step taken during the joint optimization. Evaluating the Kernel Function **_K_** is time consuming, so SMO approximates the step size by the absolute value of **\|E1−E2\|**.

*   If **_E1_** is positive, then SMO chooses an example with minimum error **_E2_**.
*   If **_E1_** is negative, SMO chooses an example with maximum error **_E2_**.

If two training examples share an identical input vector(**_x_**), then a positive progress cannot be made because the objective function becomes semi-definite. In this case, SMO uses a hierarchy of second choice heuristics until it finds a pair of Lagrange multipliers that can make a positive progress. The hierarchy of second choice heuristics is as follows :

*   SMO starts iterating through the non-bound examples, searching for a second example that can make a positive progress.
*   If none of the non-bound examples makes a positive progress, then SMO starts iterating through the entire training set until an example is found that makes positive progress.

Both the iteration through the non-bound examples and the iteration through entire training set are started at random locations so as to not bias SMO. In situations when none of the examples will make an adequate second example, then the first example is skipped and SMO continues with another chosen first example. However, this situation is very rare.

#### Computing the threshold :

After optimizing **_αi_** and **_αj_**, we select the threshold **_b_** such that the KKT conditions are satisfied for the _ith_ and _jth_ examples. When the new **_α1_** is not at the bounds, then the output of SVM is forced to be **_y1_** when the input is **_x1_**, hence the following threshold(**_b1_**) is valid :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161929/SMO/1__9JMphKPMwC021i3ClRdUQ_ys4voa.png" /></div>


When the new **_α2_** is not at the bounds, then the output of SVM is forced to be **_y2_** when the input is **_x2_**, hence the following threshold(**_b2_**) is valid :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595161969/SMO/1_VRJc7C6EIWQUxFRbip1qHA_lepxvv.png" /></div>



When both **_b1_** and **_b2_** are valid, they are equal. When both new Lagrange multipliers are at bound and if L is not equal to H, then all the thresholds between the interval **_b1_** and **_b2_** satisfy the KKT conditions. SMO chooses the threshold to be halfway in between **_b1_** and **_b2_**. The complete equation for **_b_** is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595162004/SMO/1_Wf80-UjEzie9PQ4_y5xSnw_yt2fac.png" /></div>


Now to compute Linear SVM, only a single weight vector(**_w_**) need to be stored rather than all of the training examples that correspond to non-zero Lagrange multipliers. If the joint optimization succeeds, then the stored weight vector needs to be updated to reflect the new Lagrange multiplier values. The updation step is given by :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595162060/SMO/1_YdVktYm7pqvM-tLpo_xhaA_vx387u.png" /></div>


The pseudocode for SMO is available [here](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf).

### **References :**

*   [https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)
*   [https://en.wikipedia.org/wiki/Sequential\_minimal\_optimization](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)
*   [http://pages.cs.wisc.edu/~dpage/cs760/SMOlecture.pdf](http://pages.cs.wisc.edu/~dpage/cs760/SMOlecture.pdf)
*   [http://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf](http://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf)
*   [http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/Slides5A.pdf](http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/Slides5A.pdf)
*   [http://cs229.stanford.edu/materials/smo.pdf](http://cs229.stanford.edu/materials/smo.pdf)
*   [http://cs229.stanford.edu/notes/cs229-notes3.pdf](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
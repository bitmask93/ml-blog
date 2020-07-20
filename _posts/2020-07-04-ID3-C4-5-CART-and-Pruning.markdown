---
layout: post
title: ID3, C4.5, CART and Pruning
date: 2020-07-04 13:32:20 +0300
description: In this blog, I will discuss about some algorithms used while Training Decision Trees.
tags: [Data Science, Machine Learning, Math]
---

#### In this blog, I will discuss about some algorithms used while Training Decision Trees.

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595213602/DT/0_9Pp_FKINCLcGiLE1_fojjny.jpg" /></div>

<div style="text-align:center"><p>Photo by <a href="https://unsplash.com/@pietrozj?utm_source=medium&amp;utm_medium=referral">Pietro Jeng</a> on <a href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral">Unsplash</a></p></div>

### Introduction

Decision Trees are Machine Learning algorithms that is used for both classification and Regression. Decision Trees can be used for multi-class classification tasks also. Decision Trees use a Tree like structure for making predictions where each internal nodes represents the _test_(if attribute A takes vale <5) on an attribute and each branch represent the outcome of the test on the attribute. The leaf nodes represent the class label. Some of the popular algorithms that are used to generate a Decision tree from a Dataset are ID3, c4.5 and CART.

### ID3 Algorithm

ID3 stands for Iterative Dichotomiser 3 which is a learning algorithm for Decision Tree introduced by Quinlan Ross in 1986. ID3 is an iterative algorithm where a subset(window) of the training set is chosen at random to build a decision tree. This tree will classify every objects within this window correctly. For all the other objects that are not there in the window, the tree tries to classify them and if the tree gives correct answer for all these objects then the algorithm terminates. If not, then the incorrectly classified objects are added to the window and the process continues. This process continues till a correct Decision Tree is found. This method is fast and it finds the correct Decision Tree in a few iterations. Consider an arbitrary collection of _C_ objects. If _C_ is empty or contains only objects of a single class, then the Decision Tree will be a simple tree with just a leaf node labelled with that class. Else consider _T_ to be test on an object with outcomes{_O₁, O₂, O₃….Ow_}. Each Object in _C_ will give one of these Outcomes for the test _T_. Test _T_ will partition _C_ into {_C₁, C₂, C₃….Cw_}. where **Cᵢ** contains objects having outcomes **Oᵢ**. We can visualize this with the following diagram :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595213744/DT/1_ANJMNuzOo3sxROCSuKQjgw_rfexvb.png" /></div>

<div style="text-align:center"><p><a href="https://hunch.net/~coms-4771/quinlan.pdf">Source</a></p>
</div>



When we replace each individual **Cᵢ** in the above figure with a Decision Tree for **Cᵢ**, we would get a Decision tree for all the _C_. This is a divide-and-conquer strategy which will yield single-object subsets that will satisfy the one-class requirement for a leaf. So as long as we have a test which gives a non-trivial partition of any set of objects, this procedure will always produce a Decision Tree that can correctly Classify each object in _C_. For simplicity, let us consider the test to be branching on the values of an attribute, Now in ID3, For choosing the root of a tree, ID3 uses an Information based approach that depends on two assumptions. Let _C_ contain p objects of class _P_ and n of class _N_. These assumptions are :

1.  A correct decision tree for _C_ will also classify the objects in such a way that the objects will have same proportion as in _C_. The Probability that an arbitrary object will belong to class _P_ is given below as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214046/DT/1_fiAYaMShvG6x8plrmJHkMQ_jlp3gg.gif" /></div>

And the probability that it will belong to class _N_ is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214087/DT/1_1Wd5I5ApZJhRKCTsVDsFkQ_nslcs0.gif" /></div>

2\. A decision tree returns a class to which an object belongs to. So a decision tree can be considered as a source of a message _P_ or _N_ and the expected information needed to generate this message is given as :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214183/DT/1_zae0Ds9PjuWr4yA7JYZyKg_jhhkqj.gif" /></div>


Let us consider an attribute AA as the root with values {_A₁,A₂…..,Av_}. Now _A_ will partition _C_ into {_C₁,C₂…..,Cv_}, where _C_**_ᵢ_** has those objects in CC that have a value of _A_**_ᵢ_** of _A_. Now consider _C_**_ᵢ_** having _p_**_ᵢ_** objects of class _P_ and _ni_ objects of class _N_. The expected information required for the subtree for _C_**_ᵢ_** is _I(p_**_ᵢ_**_,n_**_ᵢ_**_)_. The expected information required for the tree with _A_ as root is obtained by :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214229/DT/1_3r3a_TqrGqGG62YhCiXUCA_rqyue4.gif" /></div>

Now this is a weighted average where the weight for the _i_th branch is the proportion of Objects in _C_ that belong to _C_**_ᵢ_**. Now the information that is gained by selecting _A_ as root is given by :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214265/DT/1_Yx8G_mKcxvL13-owDGAzAQ_uibirk.gif" /></div>


Here _I_ is called the Entropy. So here ID3 choose that attribute to branch for which there is maximum Information Gain. So ID3 examines all the attributes and selects that _A_ which maximizes the _gain(A)_ and then uses the same process recursively to form Decision Trees for the subsets {_C₁,C₂…..,Cv_} till all the instances within a branch belong to same class.

**Drawback Of Information Gain**

Information gain is biased towards test with many occurances. Consider a feature that uniquely identifies each instance of a Training set and if we split on this feature, it would result in many brances with each branch containing instances of a single class alone(in other words pure) since we get maximum information gain and hence results in the Tree to overfit the Training set.

**Gain Ratio**

This is a modification to Information Gain to deal with the above mentioned problem. It reduces the bias towards multi-valued attributes. Consider a training dataset which contains _p_ and _n_ objects of class _P_ and _N_ respectively and the attribute _A_ has values {_A₁,A₂…..,Av_}. Let the number of objects with value _A_**_ᵢ_** of attribute _A_ be _p_**_ᵢ_** and _n_**_ᵢ_** respectively. Now we can define the Intrinsic Value(IV) of _A_ as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214299/DT/1_MN7zpL7W9ZlvEoq-a2u7pg_ytow5p.gif" /></div>

_IV(A)_ measures the information content of the value of Attribute _A_. Now the Gain Ratio or the Information Gain Ratio is defined as the ratio between the Information Gain and the Intrinsic Value.

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214339/DT/1_WJg-ukXiATIJm-ndSv_PXA_tli0dr.gif" /></div>


Now here we try to pick an Attribute for which the Gain Ratio is as large as possible. This ratio may not be defined when _IV(A)_ = 0. Also gain ratio may tend to favour those attributes for which the Intrinsic Value is very small. When all the attributes are Binary, the gain ratio criteria has been found to produce smaller trees.

### C4.5 Algorithm

This is another algorithm that is used to create a Decision Tree. This is an extension to ID3 algorithm. Given a training dataset _S_ = _S₁,S₂,…._ C4.5 grows the initial tree using the divide-and-conquer approach as :

*   If all the instances in _S_ belongs to the same class, or if _S_ is small, then the tree is leaf and is given the label of the same class.
*   Else, choose a test based on a single attribute which has two or more outcomes. Then make the test as the root of the tree with a branch for each outcome of the test.
*   Now partition _S_ into corresponding subsets _S₁,S₂,….,_ based on the outcome of each case.
*   Now apply the procedure recursively to each of the subset _S₁,S₂,…._

Here the splitting criteria is Gain Ratio. Here the attributes can either be numeric or nominal and this determines the format of the test of the outcomes. If an attribute is numeric, then for an Attribute _A_, the test will be {_A≤h_, _A>h_}. Here _h_ is the threshold found by sorting _S_ on the values of _A_ and then choosing the split between successive values that maximizes the Gain Ratio. Here the initial tree is Pruned to avoid Overfitting by removing those branches that do not help and replacing them with leaf nodes. Unlike ID3, C4.5 handles missing values. Missing values are marked separately and are not used for calculating Information gain and Entropy.

### Classification And Regression Trees(CART)

This is a decision Tree Technique that produces either a Classification Tree when the dependent variable is categorical or a Regression Tree when the dependent variable is numeric.

**Classification Trees :**

Consider a Dataset (_D_) with features _X_ \= _x₁,x₂….,xn_ and let _y_ = _y₁,y₂…ym_ be set of all the possible classes that _X_ can take. Tree based classifiers are formed by making repetitive splits on _X_ and subsequently created subsets of _X_. For eg. _X_ could be divided such that {x\|x₃≤53.5} and {x\|x₃>53.5}. Then the first set could be divided further into X₁ = {x\|x₃≤53.5, x₁≤29.5_} and X₂={x\|x₃≤53.5, x₁>29.5} and the other set could be split into X₃ = {x\|x₃>53.5,x₁≤74.5} and X₄ = {x\|x₃>53.5, x₁>74.5}. This can be applied to problems with multiple classes also. When we divide XX into subsets, these subsets need not be divided using the same variable. ie one subset could be split based on x₁_ and other on _x₂_. Now we need to determine how to best split _X_ into subsets and how to split the subsets also. CART uses binary partition recursively to create a binary tree. There are three issues which CART addresses :

*   Identifying the Variables to create the split and determining the rule for creating the split.
*   Determine if the node of a tree is terminal node or not.
*   Assigning a predicted class to each terminal node.

**Creating Partition :**

At each step, say for an attribute _x_**_ᵢ_**, which is either numerical or ordinal, a subset of _X_ can be divided with a plane orthogonal to _xᵢ_ axis such that one of the newly created subset has x**_ᵢ_**≤s**_ᵢ_** and other has x**_ᵢ_**\>s**_ᵢ._** When an attribute _x_**_ᵢ_** is nominal and having class label belonging to a finite set _Dk_, a subset of _X_ can be divided such that one of the newly created subset has _x_**_ᵢ_** ∈ _S_**_ᵢ_**, while other has _x_**_ᵢ_** ∉ _S_**_ᵢ_** where _S_**_ᵢ_** is a proper subset of _D_**_ᵢ_**. When _D_**_ᵢ_** contains _d_ members then there are _2ᵈ−1_ splits of this form to be considered. Splits can also be done with more than one variable. Two or more continuous or ordinal variables can be involved in a _linear combination split_ in which a hyperplane which is not perpendicular to one of the axis is used to split the subset of _X_. For examples one of the subset created contains points for which _1.4x₂−10x₃≤10_ and other subset contains points for which _1.4x₂−10x₃>10_. Similarly two or more nominal values can be involved in a _Boolean Split_. For example consider two nominal variables gender and results(pass or fail) which are used to create a split. In this case one subset could contain males and females who have passed and other could contain all the males and females who have not passed.

However by using linear combination and boolean splits, the resulting tree becomes less interpretable and also the computing time is more here since the number of candidate splits are more. However by using only single variable split, the resulting tree becomes invariant to the transformations used in the variables. But while using a linear combination split, using transformations in the variables can make difference in the resulting tree. But by using linear combination split, the resulting tree contains a classifier with less number of terminal nodes, however it becomes less interpretable. So at the time of recursive partitioning, all the possible ways of splitting _X_ are considered and the one that leads to maximum purity is chosen. This can be achieved using an impurity function which gives the proportions of samples that belongs to the possible classes. One such function is called as _Gini impurity_ which is the measure of how often a randomly chosen element from a set would be incorrectly labelled if it was randomly labelled according to the distribution of labels in the subset. Let _X_ contains items belonging to _J_ classes and let _p_**_ᵢ_** be the proportion of samples labelled with class ii in the set where i∈{_1,2,3….J_}. Now _Gini impurity_ for a set of items with _J_ classes is calculated as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214383/DT/1_2rOqk03IPa8zAf7UCDCEgw_c1xnhp.gif" /></div>


So in order to select a way to split the subset of _X_ all the possible ways of splitting can be considered and the one which will result in the greatest decrease in node impurity is chosen.

**Assigning Predicted class to Terminal Node :**

To assign a class to a Terminal node a plurality rule is used : ie the class that is assigned to a terminal node is the class that has largest number of samples in that node. If there is a node where there is a tie in two or more classes for having largest number of samples, then if a new datapoint _x_ belongs to that node, then the prediction is arbitrarily selected from among these classes.

**Determining Right size of Tree :**

The trickiest part of creating a Decision Tree is choosing the right size for the Tree. If we keep on creating nodes, then the tree becomes complex and it will result in the resulting Decision Tree created to Overfit. On the other hand, if the tree contains only a few terminal nodes, then the resulting tree created is not using enough information in the training sample to make predictions and this will lead to Underfitting. Inorder to determine the right size of the tree, we can keep an independent test sample, which is a collection of examples that comes from the same population or same distribution as the training set but not used for training the model. Now for this test set, misclassification rate is calculated, which is the proportion of cases in the test set that are misclassified when predicted classes are obtained using the tree created from the training set. Now initially when a tree is being created, the misclassification rate for the test starts to reduce as more nodes are added to it, but after some point, the misclassification rate for the test set will start to get worse as the tree becomes more complex. We could also use Cross-Validation to estimate the misclassification rate. Now the question is how to grow a best tree or how to create a set of candidate keys from which the best one can be selected based on the estimated misclassification rates. So one method to do this is to grow a very large tree by splitting subsets in the current partition of _X_ even if the split doesn’t lead to appreciable decrease in impurity. Now by using pruning, a finite sequence of smaller trees can be generated, where in the pruning process the splits that were made are removed and a tree having a fewer number of nodes is produced. Now in the sequence of trees, the first tree produced by pruning will be a subtree of the original tree, and a second pruning step creates a subtree of the first subtree and so on. Now for each of these trees, misclassification rate is calculated and compared and the best performing tree in the sequence is chosen as the final classifier.

**Regression Trees :**

CART creates regression trees the same way it creates a tree for classification but with some differences. Here for each terminal node, instead of assigning a class a numerical value is assigned which is computed by taking the sample mean or sample median of the response values for the training samples corresponding to the node. Here during the tree growing process, the split selected at each stage is the one that leads to the greatest reduction in the sum of absolute differences between the response values for the training samples corresponding to a particular node and their sample median. The sum of square or absolute differences is also used for tree pruning.

### Decision Tree Pruning

There are two techniques for pruning a decision tree they are : pre-pruning and post-pruning.

**Post-pruning**

In this a Decision Tree is generated first and then non-significant branches are removed so as to reduce the misclassification ratio. This can be done by either converting the tree to a set of rules or the decision tree can be retained but replace some of its subtrees by leaf nodes. There are various methods of pruning a tree. Here I will discuss some of them.

*   **Reduced Error Pruning(REP)**

This is introduced by Quinlan in 1987 and this is one of the simplest pruning strategies. However in practical Decision Tree pruning REP is seldom used since it requires a separate set of examples for pruning. In REP each node is considered a candidate for pruning. The available data is divided into 3 sets : one set for training(train set), the other for pruning(validation set) and a set for testing(test set). Here a subtree can be replaced by leaf node when the resultant tree performs no worse than the original tree for the validation set. Here the pruning is done iteratively till further pruning is harmful. This method is very effective if the dataset is large enough.

*   **Error-Complexity Pruning**

In this a series of trees pruned by different amounts are generated and then by examining the number of misclassifications one of these trees is selected. While pruning, this method takes into account of both the errors as well as the complexity of the tree. Before the pruning process, each leaves will contain only examples which belong to one class, as pruning progresses the leaves will include examples which are from different classes and the leaf is allocated the class which occurs most frequently. Then the error rate is calculated as the proportion of training examples that do not belong to that class. When the sub-tree is pruned, the expected error rate is that of the starting node of the sub-tree since it becomes a leaf node after pruning. When a sub-tree is not pruned then the error rate is the average of error rates at the leaves weighted by the number of examples at each leaf. Pruning will give rise to an increase in the error rate and dividing this error rate by number of leaves in the sub-tree gives a measure of the reduction in error per leaf for that sub-tree. This is the error-rate complexity measure. The error cost of node _t_ is given by :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214433/DT/1_gLQgqYudUxwEeIcb2NnkBw_thktd9.gif" /></div>


_r(t)_ is the error rate of a node which is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214477/DT/1_uKgNOrJyKnH-y-V77LT2lw_oxcjtu.gif" /></div>


_p(t)_ is the proportion of data at node t which is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214509/DT/1_WZYIMZahCk-cQnmtZi1VHw_z3lbff.gif" /></div>

When a node is not pruned, the error cost for the sub-tree is :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214544/DT/1_8DF2zCof4KOuVK14WzU-1w_agoh4m.gif" /></div>


The complexity cost is the cost of one extra leaf in the tree which is given as _α_. Then the total cost of the sub-tree is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214574/DT/1_jKpVC4LBhZq-OP9a9Tu_2w_i8htzt.gif" /></div>

The cost of a node when pruning is done is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214613/DT/1_aTHqw6I8y7yrYv8Ch-NUPg_dgoebv.gif" /></div>


Now when these two are equal ie :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214648/DT/1__UWNKoxnGR-4SvNpatGfZQ_uksqs6.gif" /></div>

_α_ gives the reduction in error per leaf. So the algorithm first computes αα for each sub-tree except the first and then selects that sub-tree that has the smallest value of αα for pruning. This process is repeated till there are no sub-trees left and this will yield a series of increasingly pruned trees. Now the final tree is chosen that has the lowest misclassification rate for this we need to use an independent test data set. According to Brieman’s method, the smallest tree with a mis-classification within one standard error of the minimum mis-classification rate us chosen as the final tree. The standard error of mis-classification rate is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214713/DT/1_RbhZPABqbxYC_5uzoaLlqg_qwrktl.gif" /></div>

Where _R_ is the mis-classification rate of the Pruned tree and _N_ is the number of examples in the test set.

*   **Minimum-Error Pruning**

This method is used to find a single tree that minimizes the error rate while classifying independent sets of data. Consider a dataset with _k_ classes and nn examples of which the greatest number(_nₑ_) belong to class _e_. Now if the tree predicts class _e_ for all the future examples, then the expected error rate of pruning at a node assuming that each class is equally likely is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214753/DT/1_b3bGL-BMKkAM7-a51yrpfA_nswk2w.gif" /></div>

Where _R_ is the mis-classification rate of the Pruned tree and _N_ is the number of examples in the test set.

Now for each node in the tree, calculate the expected error rate if that sub-tree is pruned. Now calculate the expected error rate if the node is not pruned. Now do the process recursively for each node and if pruning the node leads to increase in expected error rate, then keep the sub-tree otherwise prune it. The final tree obtained will be pruned tree that minimizes the expected error rate in classifying the independent data.

**Pre-pruning**

This is a method that is used to control the development of a decision tree by removing the non-significant nodes. This is a top-down approach. Pre-pruning is not exactly a “pruning” technique since it does not prune the branches of an existing tree. They only suppress the growth of a tree if addition of branches does not improve the performance of the overall.

**Chi-square pruning**

Here a statistical test(chi-square test) is applied to determine if the split on a feature is statistically significant. Here the null hypothesis is that the actual and predicted values are independent and then a significant test is used to determine if the null hypothesis can be accepted or not. The significant test computes the probability that the same or more extreme value of the statistic will occur by chance if the null hypothesis is correct. This is called the _p−value_ of the test and if this value is too low, null hypothesis can be rejected. For this the observed _p−value_ is compared with that of a significance level αα which is fixed. While pruning a decision tree, rejecting null hypothesis means retaining a subtree instead of pruning it. So first a contingency table is created, which is used to summarize the relationship between several categorical variables. The structure of a contingency table is given as below :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214792/DT/1_fDlAhWsn8xqs1y8uIf0HwA_fmi9ak.png" /></div>

<div style="text-align:center"><p><a href="https://www.cs.waikato.ac.nz/~eibe/pubs/thesis.final.pdf">Source</a></p></div>


Here the rows and columns correspond to the values of the nominal attribute :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214921/DT/1_H-fIPaffr1-ll6RPT9XSEA_qil1xy.gif" /></div>


Now the chi-squared test statistic can be calculated using :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595214961/DT/1_WMs6lre7BtxIVcSeOLCgNw_slvque.gif" /></div>


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595215001/DT/1__-ibVoQ-Ceney-n40NLiBw_moyibl.gif" /></div>


Where :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595215035/DT/1_h-50PMgQWGy9TeJqBhrdIg_sbefmg.gif" /></div>

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595215068/DT/1_gfZBQpjhvJHIHzvr4u5big_xlzndq.gif" /></div>


Under Null Hypothesis these probabilities are independent and so the product of these two probabilities will be the probability that an observation will fall into cell _(i , j)_. Now consider an attribute _A_ and under null hypothesis _A_ is independent of Class objects. Now using the chi-squared test statistic, we can determine the confidence with which we can reject the null hypothesis ie we retain a subtree instead of pruning it. If 𝜒² value is greater than a threshold(_t_), then the information gain due to the split is significant. So we keep the sub-tree, and if 𝜒² value is less than the threshold(_t_), then the information gained due to the split is less significant and we can prune the sub-tree.

### References

The following contains links to the works which helped me while writing this blog. So this work is just a summary of all the below mentioned works. For further reading on all the above mentioned, please do have a look at the following :

*   [https://link.springer.com/content/pdf/10.1007/BF00116251.pdf](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf)
*   [https://www.researchgate.net/publication/324941161\_A\_Survey\_on\_Decision\_Tree\_Algorithms\_of\_Classification\_in\_Data\_Mining](https://www.researchgate.net/publication/324941161_A_Survey_on_Decision_Tree_Algorithms_of_Classification_in_Data_Mining)
*   [http://mas.cs.umass.edu/classes/cs683/lectures-2010/Lec23\_Learning2-F2010-4up.pdf](http://mas.cs.umass.edu/classes/cs683/lectures-2010/Lec23_Learning2-F2010-4up.pdf)
*   [https://en.wikipedia.org/wiki/Information\_gain\_ratio#cite\_note-2](https://en.wikipedia.org/wiki/Information_gain_ratio#cite_note-2)
*   [http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf](http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf)
*   [http://www.cs.umd.edu/~samir/498/10Algorithms-08.pdf](http://www.cs.umd.edu/~samir/498/10Algorithms-08.pdf)
*   [https://en.wikipedia.org/wiki/C4.5\_algorithm](https://en.wikipedia.org/wiki/C4.5_algorithm)
*   [https://en.wikipedia.org/wiki/Information\_gain\_in\_decision\_trees#cite\_note-1](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#cite_note-1)
*   [https://en.wikipedia.org/wiki/Decision\_tree\_learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
*   [https://en.wikipedia.org/wiki/Predictive\_analytics#Classification\_and\_regression\_trees\_.28CART.29](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)
*   [http://mason.gmu.edu/~csutton/vt6.pdf](http://mason.gmu.edu/~csutton/vt6.pdf)
*   [https://en.wikipedia.org/wiki/Decision\_tree\_learning#cite\_note-bfos-6](https://en.wikipedia.org/wiki/Decision_tree_learning#cite_note-bfos-6)
*   [https://pdfs.semanticscholar.org/025b/8c109c38dc115024e97eb0ede5ea873fffdb.pdf](https://pdfs.semanticscholar.org/025b/8c109c38dc115024e97eb0ede5ea873fffdb.pdf)
*   [https://arxiv.org/pdf/1106.0668.pdf](https://arxiv.org/pdf/1106.0668.pdf)
*   [https://link.springer.com/content/pdf/10.1023/A:1022604100933.pdf](https://link.springer.com/content/pdf/10.1023/A:1022604100933.pdf)
*   [https://www.cs.waikato.ac.nz/~eibe/pubs/thesis.final.pdf](https://www.cs.waikato.ac.nz/~eibe/pubs/thesis.final.pdf)
*   [https://hunch.net/~coms-4771/quinlan.pdf](https://hunch.net/~coms-4771/quinlan.pdf)
*   [https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs2.pdf](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs2.pdf)

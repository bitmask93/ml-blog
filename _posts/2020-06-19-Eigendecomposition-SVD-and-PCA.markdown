---
layout: post
title: Eigendecomposition, SVD and PCA
date: 2020-06-19 13:32:20 +0300
description: In this post, I will discuss some basic linear algebra, Eigendecomposition, SVD and PCA.
tags: [Data Science, Machine Learning, Math]
---

#### In this post, I will discuss some basic linear algebra, Eigendecomposition, SVD and PCA.


![Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/2560/0*t199VnUb8tdFFVqs)*Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)*

In this article, I will discuss Eigendecomposition, Singular Value Decomposition(SVD) as well as Principal Component Analysis. Before going into these topics, I will start by discussing some basic Linear Algebra and then will go into these topics in detail.

### Basics Of Linear Algebra :

*   **Scalars**: A scalar is just a single number. Scalars are written in italics and in lower case. “Let s ∈ R be the slope of the line,” while deﬁning a real-valued scalar, or “Let n ∈ N be the number of units,” while deﬁning a natural number scalar.
*   **Vectors**: A vector is an array of numbers arranged in specific order. We can identify each individual number by its index in that ordering. A vector could be a row vector or a column vector. Typically we give vectors lowercase names in bold typeface, such as **x**. By default Vectors are Column vectors unless it’s explicitly mentioned.
*   **Matrix**: A matrix is a 2-D array of numbers, so each element is identiﬁed by two indices instead of just one. We usually give matrices uppercase variable names with bold typefaces, such as **A**. If a real-valued matrix **A** has a height of _m_ and a width of _n_, then we say that **A** ∈ R(m×n). **A**(1,1)is the upper left entry of **A** and **A**(m,n)is the bottom right entry.
*   **Tensors**: An array with more than two axes is called a Tensor.In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. We denote a tensor named “A” with this typeface: **A**. We identify the element of **A** at coordinates. (i, j, k) by writing A(i,j,k).
*   **Transpose**: The transpose of a matrix is the mirror image of the matrix across a diagonal line, called the main diagonal, running down and to the right, starting from its upper left corner. We denote the transpose of a matrix **A** as **A^**T and is defined as follows :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059008/ed-svd-pca/1_vCUiEwzado_mJ-Nim-_Y8A_rrdj8c.png" /></div>

*   Vectors can be thought of as matrices that contain only one column. The transpose of a vector is, therefore, a matrix with only one row.
*   **Addition Of Matrices**: We can add matrices to each other, as long as they have the same shape, just by adding their corresponding elements which can be defined as: **C** = **A** + **B.** Where:


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059094/ed-svd-pca/1_PEX7jJF5UoNUwU3H-6bBJg_ccy65m.png" /></div>


*   We can also add a scalar to a matrix or multiply a matrix by a scalar, just by performing that operation on each element of a matrix: **D** = a·**B**+c where:


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059183/ed-svd-pca/1_8avYwJBReN6qWSMoP4FpAA_kmqomi.png" /></div>


*   We can also do the addition of a matrix and a vector, yielding another matrix: **C**\=**A**+b, Where :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059242/ed-svd-pca/1_veV46T_enMIPc2zVLtV82A_cqthpe.png" /></div>

Here we add **_b_** to each row of the matrix. This is also called as broadcasting.

#### Multiplying Matrices and Vectors :

The matrix product of matrices **A** and **B** is a third matrix **C**. In order for this product to be deﬁned, **A** must have the same number of columns as **B** has rows. If **A** is of shape m × n and **B** is of shape n × p, then **C** has a shape of m × p. We can write the matrix product just by placing two or more matrices together :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059407/ed-svd-pca/1_m4PMuEiqCpz6N6pliXGreQ_w69oqq.png" /></div>


The product operation is deﬁned by :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059485/ed-svd-pca/1_o3PjVF6ySzYskpkCwZOdWw_bs7xs5.png" /></div>


This is also called as the Dot Product. The following are some of the properties of Dot Product :

*   Distributive property :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059547/ed-svd-pca/1_XoODe6o6VJbEXl9xGzoD1g_keqaf2.png" /></div>


*   Associativity :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059601/ed-svd-pca/1_YsS0liGyd5iqhOG-WD1oBA_av3zza.png" /></div>

*   Dot Product is not Commutative :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059853/ed-svd-pca/CodeCogsEqn_2_e3xi2g.gif" /></div>


**Identity Matrix**: An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix. All the entries along the main diagonal are 1, while all the other entries are zero.

**Inverse of a Matrix**: The matrix inverse of A is denoted as A^(−1), and it is deﬁned as the matrix such that :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059911/ed-svd-pca/1_jwc0Xbw8LjlRhhVuNZWsxQ_zv9jn2.png" /></div>


This can be used to solve a system of linear equations of the type **_Ax = b_** where we want to solve for **_x_**:


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595059964/ed-svd-pca/1_0euIwX-nZIkd7vcghTyKmg_nrhru5.png" /></div>


#### Linear independence and Span :

A set of vectors is **linearly independent** if no vector in a set of vectors is a linear combination of the other vectors. The **span** of a set of vectors is the set of all the points obtainable by linear combination of the original vectors.

#### Norms :

Used to measure the size of a vector. Formally the **_Lp_** norm is given by:

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595060025/ed-svd-pca/1_bMbZQvTwK8fuAl8CwrtRIQ_mvo9d5.png" /></div>


On an intuitive level, the norm of a vector **_x_** measures the distance from the origin to the point **_x_**. The L² norm, with p = 2, is known as the **Euclidean norm**, which is simply the Euclidean distance from the origin to the point identiﬁed by **_x_**. The L² norm is often denoted simply as \|\|x\|\|,with the subscript 2 omitted. It is also common to measure the size of a vector using the squared L² norm, which can be calculated simply as :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595060077/ed-svd-pca/1_X7kb5i9ytZ6EH3cHDiUU8g_uwasue.png" /></div>

The squared L² norm is more convenient to work with mathematically and computationally than the L² norm itself. In many contexts, the squared L² norm may be undesirable because it increases very slowly near the origin. In these cases, we turn to a function that grows at the same rate in all locations, but that retains mathematical simplicity: the L¹ norm :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595060137/ed-svd-pca/1_E6jrdwxcaklZvMLoUwTNkA_yubeo1.png" /></div>

The L¹ norm is commonly used in machine learning when the diﬀerence between zero and nonzero elements is very important.

**Frobenius norm**: Used to measure the size of a matrix. Also called Euclidean norm (also used for vector L². norm) :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595060208/ed-svd-pca/1_HtYAezUS9MU8Joh9PAmpug_fbmr42.png" /></div>

It is also equal to the square root of the matrix trace of AA^(H), where A^(H) is the conjugate transpose :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595060265/ed-svd-pca/1_KnZ5spEIsTxEBFxgCAFJzQ_ds6zco.png" /></div>


Trace of a square matrix **A** is defined to be the sum of elements on the main diagonal of **A**. The trace of a matrix is the sum of its eigenvalues, and it is invariant with respect to a change of basis.

#### EigenValues And EigenVectors:

An eigenvector of a square matrix **A** is a nonzero vector **_v_** such that multiplication by **_A_** alters only the scale of **_v_** and not the direction :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595060940/ed-svd-pca/1_KarSOTXz9ZQHkVBqRqZyxA_vtfweb.png" /></div>


The scalar **_λ_** is known as the eigenvalue corresponding to this eigenvector. The vector **Av** is the vector **v** transformed by the matrix **A**. This transformed vector is a scaled version (scaled by the value **λ**) of the initial vector **v**. If **v** is an eigenvector of **A**, then so is any rescaled vector **sv** for **s ∈ R**, **s != 0**. Moreover, **sv** still has the same eigenvalue. Consider the following vector(**v**) :
{% highlight ruby %}
import numpy as np  
v = np.array([[1], [1]])
{% endhighlight  %}
Let’s plot this vector and it looks like the following :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595061077/ed-svd-pca/1_ry13GiHqg3A68nx_-fcekw_h5iy04.png" /></div>

<div style="text-align:center">Image by Author</div>

Now consider the matrix(**A**):
{% highlight ruby %}
A = np.array([[5, 1], [3, 3]])
{% endhighlight  %}
Now let’s take the dot product of **A** and **v** and plot the result, it looks like the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595061381/ed-svd-pca/1_mq-fBFsUiNhPwJKc_2CVbw_gsbsuf.png" /></div>

<div style="text-align:center">Image by Author</div>


Here, the blue vector is the original vector(**v**) and the orange is the vector obtained by the dot product between **v** and **A**. Here we can clearly observe that the direction of both these vectors are same, however, the orange vector is just a scaled version of our original vector(**v**). So we can say that that **v** is an eigenvector of **A**. eigenvectors are those Vectors(**v**) when we apply a square matrix **A** on **v**, will lie in the same direction as that of **v**.

### Eigendecomposition :

Suppose that a matrix **A** has n linearly independent eigenvectors **{v1,….,vn}** with corresponding eigenvalues **{λ1,….,λn}**. We can concatenate all the eigenvectors to form a matrix **V** with one eigenvector per column likewise concatenate all the eigenvalues to form a vector **λ**. The Eigendecomposition of **A** is then given by :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595063337/ed-svd-pca/1_3u0ZYzA-GCkqURa7l3CdQw_i1fe0p.png" /></div>


Decomposing a matrix into its corresponding eigenvalues and eigenvectors help to analyse properties of the matrix and it helps to understand the behaviour of that matrix. Say matrix **A** is real symmetric matrix, then it can be decomposed as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595063406/ed-svd-pca/1_-7BRjO7yIMorPybyHq_HCQ_ietol0.png" /></div>


where **Q** is an orthogonal matrix composed of eigenvectors of **A**, and **Λ** is a diagonal matrix. Any real symmetric matrix **A** is guaranteed to have an Eigen Decomposition, the Eigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue, and we could equivalently choose a **Q** using those eigenvectors instead.

**What does Eigendecomposition tell us?**

*   The matrix is singular(**\|A\|=0**) if and only if any of the eigenvalues are zero. The Eigendecomposition of a real symmetric matrix can also be used to optimize quadratic expressions of the form :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595063484/ed-svd-pca/1_aw42GG027iYxE9QuuLApSw_regct9.png" /></div>


*   Whenever **_x_** is equal to an eigenvector of **_A_**, **_f_** takes on the value of the corresponding eigenvalue. The maximum value of f within the constraint region is the maximum eigenvalue and its minimum value within the constraint region is the minimum eigenvalue.
*   A matrix whose eigenvalues are all positive is called **positive deﬁnite**. A matrix whose eigenvalues are all positive or zero valued is called **positive semideﬁnite**. If all eigenvalues are negative, the matrix is **negative deﬁnite**. If all eigenvalues are negative or zero valued, it is **negative semideﬁnite**.
*   Positive semideﬁnite matrices are guarantee that :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595063556/ed-svd-pca/1_kRNXeWF2tNyqTi-U5YEJ0g_f92yck.png" /></div>


*   Positive deﬁnite matrices additionally guarantee that :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595063623/ed-svd-pca/1_NjtY5PtU6B27Go04CrcYzA_arixhg.png" /></div>


### Singular Value Decomposition

Singular Value Decomposition(SVD) is a way to factorize a matrix, into **singular vectors** and **singular values**. A **singular matrix** is a square matrix which is not invertible. Alternatively, a matrix is singular if and only if it has a determinant of 0. The **singular values** are the absolute values of the eigenvalues of a matrix **A.** SVD enables us to discover some of the same kind of information as the eigen decomposition reveals, however, the SVD is more generally applicable. Every real matrix has a singular value decomposition, but the same is not true of the eigenvalue decomposition. The singular value decomposition is similar to Eigen Decomposition except this time we will write **A** as a product of three matrices :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595063694/ed-svd-pca/1_6Lh2oK2uFggS76vChET7FQ_dsqgri.png" /></div>


**U** and **V** are orthogonal matrices. **D** is a diagonal matrix (all values are 0 except the diagonal) and need not be square. The columns of **U** are called the left-singular vectors of **A** while the columns of **V** are the right-singular vectors of **A**. The values along the diagonal of **D** are the singular values of **A**. Suppose that **A** is an _m ×n_ matrix, then **U** is deﬁned to be an _m × m_ matrix, **D** to be an _m × n_ matrix, and **V** to be an n × n matrix.

The intuition behind SVD is that the matrix **A** can be seen as a linear transformation. This transformation can be decomposed in three sub-transformations: 1. rotation, 2. re-scaling, 3. rotation. These three steps correspond to the three matrices **U**, **D**, and **V**. Now let’s check if the three transformations given by the SVD are equivalent to the transformation done with the original matrix.

Consider a unit circle given below :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595064208/ed-svd-pca/1_oUM5U10CGC9WOQ8hKBty2g_wrqwq0.png" /></div>

<div style="text-align:center">Image by Author</div>


Here the red and green are the basis vectors. Not let us consider the following matrix **A** :
{% highlight ruby %}
A = np.array([[3 , 7],[5, 2]])
{% endhighlight %}
Applying the matrix **A** on this unit circle, we get the following :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595064518/ed-svd-pca/1_w6fLXbblS_5-GDt0YVYKug_nwwzae.png" /></div>

<div style="text-align:center">Image by Author</div>


Now let us compute the SVD of matrix **A** and then apply individual transformations to the unit circle:
{% highlight ruby %}
U, D, V = np.linalg.svd(A)
{% endhighlight %}
Now applying U to the unit circle we get the First Rotation :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595064854/ed-svd-pca/1_CjrlBlsivWntp42WLSsFTw_l2d9do.png" /></div>

<div style="text-align:center">Image by Author</div>

Now applying the diagonal matrix **D** we obtain a scaled version on the circle :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595064932/ed-svd-pca/1_Hd5Qbks2tUDuQrBIs8ZSEQ_qskxvk.png" /></div>


<div style="text-align:center">Image by Author</div>


Now applying the last rotation(**V**), we obtain the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595065023/ed-svd-pca/1_rQcUQFA4g8W2u4PKWSnOug_vw7lk4.png" /></div>

<div style="text-align:center">Image by Author</div>


Now we can clearly see that this is exactly same as what we obtained when applying **A** directly to the unit circle. Singular Values are ordered in descending order. They correspond to a new set of features (that are a linear combination of the original features) with the first feature explaining most of the variance. To find the sub-transformations :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595065099/ed-svd-pca/1_ax9sce5_pTTKY2G3i9FJdg_qvf4fm.png" /></div>


#### Truncated SVD :

Now we can choose to keep only the first r columns of **U**, r columns of **V** and r×r sub-matrix of **D** ie instead of taking all the singular values, and their corresponding left and right singular vectors, we only take the _r_ largest singular values and their corresponding vectors. We form an approximation to **A** by truncating, hence this is called as _Truncated SVD_. How to choose _r_? If we choose a higher _r_, we get a closer approximation to A. On the other hand, choosing a smaller _r_ will result in loss of more information. So we need to choose the value of _r_ in such a way that we can preserve more information in **A**. One way pick the value of _r_ is to plot the log of the singular values(diagonal values **σ**) and number of components and we will expect to see an elbow in the graph and use that to pick the value for _r_. This is shown in the following diagram :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595065867/ed-svd-pca/1_R-gx9OopWi2dTun6HiD9DA_c73xsy.png" /></div>

<div style="text-align:center">Image by Author</div>

However, this does not work unless we get a clear drop-off in the singular values. In real-world we don’t obtain plots like the above. Most of the time when we plot the log of singular values against the number of components, we obtain a plot similar to the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595070457/ed-svd-pca/1_exYVyXAie9Q5D3D04IvUzw_mwd9h2.png" /></div>


<div style="text-align:center">Image by Author</div>

What do we do in case of the above situation? We can use the ideas from the paper by [Gavish and Donoho](https://arxiv.org/pdf/1305.5870.pdf) on optimal hard thresholding for singular values. Their entire premise is that our data matrix **A** can be expressed as a sum of two low rank data signals :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595070656/ed-svd-pca/1_63mbHGPGefkvsOVLTsOPSw_ezq9s6.png" /></div>


Here the fundamental assumption is that :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595070741/ed-svd-pca/1_Zdi4aVaYKbwX3LcMmIbCJw_gsa13f.png" /></div>


That is noise has a Normal distribution with mean 0 and variance 1. If Data has low rank structure(ie we use a cost function to measure the fit between the given data and its approximation) and a Gaussian Noise added to it, We find the first singular value which is larger than the largest singular value of the noise matrix and we keep all those values and truncate the rest.

**Case 1(Best Case Scenario) :**

**A** is a Square Matrix and **γ** is known. Here we truncate all **σ<τ**(Threshold). The Threshold **τ** can be found using the following :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595070825/ed-svd-pca/1_UuyX-u2ap_SAvUdoZvq22w_hvw8lu.png" /></div>


**Case 2:**

**A** is a Non-square Matrix (m≠n) where m and n are dimensions of the matrix and **γ** is not known, in this case the threshold is calculated as:

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595077560/ed-svd-pca/1_MwbkDeKdWYNvzXHZOp1xUA_gpzzpz.png" /></div>


**β** is the aspect ratio of the data matrix **β**\=m/n, and :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080041/ed-svd-pca/1_TJf_wCVSA92-AXTQyTIHwQ_nc8m1k.png" /></div>



### Principal Components Analysis (PCA)

Suppose we have ‘m’ data points :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080155/ed-svd-pca/1_F0wQSav1UMzEFYhg8FGD8w_w6cqip.png" /></div>


and we wish to apply a lossy compression to these points so that we can store these points in a lesser memory but may lose some precision. So the objective is to lose as little as precision as possible. So we convert these points to a lower dimensional version such that:

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080270/ed-svd-pca/1_yWGImBL0MJWIr_PTuEVkDA_rmqluz.png" /></div>


If ‘l’ is less than n, then it requires less space for storage. We need to find an encoding function that will produce the encoded form of the input **_f(x)=c_** and a decoding function that will produce the reconstructed input given the encoded form **_x≈g(f(x))_**. The encoding function **_f(x)_** transforms **x** into **c** and the decoding function transforms back c into an approximation of **x**.

**Constraints :**

*   The decoding function has to be a simple matrix multiplication: **_g(c)=Dc_**. When we apply the matrix **D** to the Dataset of new coordinate system, we should get back the dataset in the initial coordinate system.
*   The columns of **D** must be orthogonal(**D** is semi-orthogonal **D** will be an Orthogonal matrix when ’n’ = ‘l’).
*   The columns of D must have unit norm.

#### Finding the encoding function :

We know **_g(c)=Dc_**. We will find the encoding function from the decoding function. We want to minimize the error between the decoded data point and the actual data point. That is we want to reduce the distance between **x** and **g(c)**. We can measure this distance using the L² Norm. We need to minimize the following:

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080359/ed-svd-pca/1_zQTOmtSCWVJbs7j9tB1nnQ_hup9vj.png" /></div>


We will use the Squared L² norm because both are minimized using the same value for **c**. Let **c∗** be the optimal **c**. Mathematically we can write it as :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080453/ed-svd-pca/1_vP7khfNNoQC0KaQIOaoI9Q_hug5j8.png" /></div>


But Squared L² norm can be expressed as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080552/ed-svd-pca/1_J1XL1tWU8H_RISdY_vbInA_brxjor.png" /></div>


Applying this here, we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080679/ed-svd-pca/1_rpFFUszHQ7Uea6NfDPhDJg_mu8tek.png" /></div>


By the distributive property, we get:

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080803/ed-svd-pca/1_Ho55xWF7onvIMQXhGZUghg_kxyon4.png" /></div>


Now by applying the commutative property we know that :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080907/ed-svd-pca/1_F1x5ZdvLy8qB4pq_-eLfOA_fidfiu.png" /></div>


Applying this property we now get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595080998/ed-svd-pca/1_Ifh4V5YAE5wJIji8KkRCXw_l6jptf.png" /></div>


The first term does not depend on c and since we want to minimize the function according to **c** we can just ignore this term :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081147/ed-svd-pca/1_BiosM0dQQpVTcJnq6K5xjQ_wouitq.png" /></div>


Now by Orthogonality and unit norm constraints on D :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081254/ed-svd-pca/1_78Npyoo2qmR3mNNARTve3Q_yuigrm.png" /></div>


Now we can minimize this function using Gradient Descent. The main idea is that the sign of the derivative of the function at a specific value of x tells you if you need to increase or decrease **x** to reach the minimum. When the slope is near 0, the minimum should have been reached.

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081350/ed-svd-pca/1_01MnLm0IeWJJxzLtjcfkOA_opm13p.png" /></div>


We want **c** to be a column vector of shape _(l, 1)_, so we need to take the transpose to get :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081651/ed-svd-pca/1_wERNfgWOEiucakVRHcr1pQ_tb0fy5.png" /></div>


To encode a vector, we apply the encoder function :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081556/ed-svd-pca/1__Gufh6YtauvPwh7tJmMmrQ_vgj1lf.png" /></div>


Now the reconstruction function is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081778/ed-svd-pca/1_471g0tdMWe3TgowpnGRphQ_skja6d.png" /></div>


#### Choosing the encoding matrix D

Purpose of the PCA is to change the coordinate system in order to maximize the variance along the first dimensions of the projected space. Maximizing the variance corresponds to minimizing the error of the reconstruction. Since we will use the same matrix **D** to decode all the points, we can no longer consider the points in isolation. Instead, we must minimize the Frobenius norm of the matrix of errors computed over all dimensions and all points :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081880/ed-svd-pca/1_RrQcNPV_Hjm1M9BNsO9gPw_tsw47g.png" /></div>


We will start to find only the first principal component (PC). For that reason, we will have _l = 1_. So the matrix D will have the shape (n×1). Since it is a column vector, we can call it **d**. Simplifying **D** into **d**, we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595081982/ed-svd-pca/1_FnJIu1OjMKEQBdZJLUUFVQ_hwt7i8.png" /></div>


Where **r(x)** is given as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082106/ed-svd-pca/1_iZaxn4H-AGQcXTCisp5-jg_xptbze.png" /></div>


Now plugging **r(x)** into the above equation, we get :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082236/ed-svd-pca/1_5NvhvXZHuaRccVCK24Uslg_zjqu7w.png" /></div>


We need the Transpose of _x^(i)_ in our expression of **d\***, so by taking the transpose we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082401/ed-svd-pca/1_fvrVGM4LZbOmVOP2n31-Tw_rvw0cl.png" /></div>


Now let us define a single matrix **_X_**, which is defined by stacking all the vectors describing the points such that :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082524/ed-svd-pca/1_oj7wgiT-9pgqtsbDjqVZMg_nqrc2c.png" /></div>


We can now rewrite the problem as :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082630/ed-svd-pca/1_lno4V6-88ZgYng51YDfGjg_c7agsh.png" /></div>


We can simplify the Frobenius norm portion using the Trace operator :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082761/ed-svd-pca/1_Yr83d92CkGBdzVFrLW_gjg_srpkxg.png" /></div>


Now using this in our equation for **d\***, we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082869/ed-svd-pca/1_eUt2f_u1zlzCgUQYJe1Org_ownwdz.png" /></div>


Now _Tr(AB)=Tr(BA)_. Using this we get :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595082996/ed-svd-pca/1_VscJp9q4O-P1mXQQqK5tbw_efoxvq.png" /></div>


We need to minimize for d, so we remove all the terms that do not contain d :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595083145/ed-svd-pca/1_gquykAYXVICAbentj4Qg1g_etxgkh.png" /></div>


By cyclic property of Trace :


<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595083289/ed-svd-pca/1_ZM9K_eGVuM9JRv4mSf0tvA_pcoexq.png" /></div>


By applying this property, we can write **d\*** as :

<div style="text-align:center"><img src="https://res.cloudinary.com/jithinjayan1993/image/upload/v1595083398/ed-svd-pca/1_4Bx_uXXLIh7jEhq6HDge1Q_cugiwc.png" /></div>


We can solve this using eigendecomposition. The optimal d is given by the eigenvector of **_X^(T)X_** corresponding to largest eigenvalue. This derivation is specific to the case of _l=1_ and recovers only the first principal component. The matrix **_X^(T)X_** is called the Covariance Matrix when we centre the data around 0. The covariance matrix is a _n ⨉ n_ matrix. Its diagonal is the variance of the corresponding dimensions and other cells are the Covariance between the two corresponding dimensions, which tells us the amount of redundancy. This means that larger the covariance we have between two dimensions, the more redundancy exists between these dimensions. That means if variance is high, then we get small errors. To maximize the variance and minimize the covariance (in order to de-correlate the dimensions) means that the ideal covariance matrix is a diagonal matrix (non-zero values in the diagonal only).The diagonalization of the covariance matrix will give us the optimal solution.

You can find more about this topic with some examples in python in my Github repo, click [here](https://github.com/bitmask93/Linear_Models/blob/master/4-Eigendecomposition-SVD-PCA.ipynb).

### References :

*   [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) — Deep Learning book by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
*   [https://arxiv.org/pdf/1305.5870.pdf](https://arxiv.org/pdf/1305.5870.pdf)
*   [https://arxiv.org/pdf/1404.1100.pdf](https://arxiv.org/pdf/1404.1100.pdf)
*   [https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/)
*   [https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.12-Example-Principal-Components-Analysis/](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.12-Example-Principal-Components-Analysis/)
*   [https://brilliant.org/wiki/principal-component-analysis/#from-approximate-equality-to-minimizing-function](https://brilliant.org/wiki/principal-component-analysis/#from-approximate-equality-to-minimizing-function)
*   [https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.7-Eigendecomposition/](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.7-Eigendecomposition/)
*   [http://infolab.stanford.edu/pub/cstr/reports/na/m/86/36/NA-M-86-36.pdf](http://infolab.stanford.edu/pub/cstr/reports/na/m/86/36/NA-M-86-36.pdf)
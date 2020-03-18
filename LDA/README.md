<h1 align=center>Lda.h</h2>
<h6 align=center>Author: Jachin Fang</h6>
</br>
</br>

The Data Reduction Library <Lda.h> header.

<blockquote>
&emsp; Linear Discriminant Analysis (LDA) is most commonly used as dimensionality reduction technique in the pre-processing step for
// pattern-classification and machine learning applications. The goal is to project a dataset onto a lower-dimensional space
// with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs.
</blockquote>
</br>

##### How to use

For example, here is some data:
```c++
vector<double> data{
    4,2, // class 1
    2,4,
    2,3,
    3,6,
    4,4,

    9,10, // class 2
    6,8,
    9,5,
    8,7,
    10,8
};
vector<int> labels{ 1,1,1,1,1, 2,2,2,2,2 };
unsigned int dim = 2;
unsigned int size = 10;
``` 

</br>

Then

```c++
#include <Lda.h>
``` 

```c++
Lda<double, int> lda;
vector<double> result = lda.reduce(data, dim, labels, size)[1];
```

Or
```c++
Lda<double, int> lda;
lda.reduce(data, dim, labels, size);
vector<double> result = lda[1];
```


<h1 align=center>Pca.h</h2>
<h6 align=center>Author: Jachin Fang</h6>
</br>
</br>

The Data Reduction Library <Pca.h> header.

<blockquote>
&emsp; Principal Component Analysis (PCA) is a statistical procedure that orthogonally transforms
the original `N` coordinates of a data set into a new set of `K` coordinates called principal components. (K < N)
</blockquote>
</br>

##### How to use

For example, here is some data:
```c++
vector<double> data{
    10.2352, 11.3220,
    10.1223, 11.8110,
    9.1902, 8.9049,
    9.3064, 9.8474,
    8.3301, 8.3404,
    10.1528, 10.1235,
    10.4085, 10.8220,
    9.0036, 10.0392,
    9.5349, 10.0970,
    9.4982, 10.8254
};
unsigned int dim = 2;
unsigned int size = 10;
``` 

And here is test data:
```c++
vector<double> test{ 10, 202 };
```

</br>

Then

```c++
#include <Pca.h>
``` 

```c++
Pca<float> pca;
// transforms `2` coordinates of a data into a new set of `1` coordinates.
vector<double> result = pca.reduce(data, dim, size)[1]; 
```

Or
```c++
Pca<float> pca;
// transforms `2` coordinates of a data into a new set of `1` coordinates.
pca.reduce(data, dim, size);
vector<double> result = pca[1]; 
```

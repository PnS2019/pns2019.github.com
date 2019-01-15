---
layout: default
---

Welcome to the first session of the module ['Deep Learning on Raspberry Pi'](../README.md).

In this session, we will revisit the basic concepts of linear algebra. Then, we will familiarize ourselves with Numpy, a Python package used for scientific computing and some basics of symbolic computation. At the end of this session, there will be a few exercises which will further help the understanding of the concepts introduced here.

## Linear Algebra

This section will only provide a brief introduction to linear algebra. For those of you who are unfamiliar with the concepts of linear algebra, it is strongly recommended that you spend some time with a text book or a complete course on linear algebra. A strongly recommended text book is [Introduction to Linear Algebra](math.mit.edu/~gs/linearalgebra/) by Gilbert Strang.

For those of you who are well familiar with linear algebra, you may skip the next two sections.

### Vector Space
A vector space $$ V $$, over the set of real numbers $$ \mathbb{R} $$, is a set equipped with two operations, addition `+` and multiplication `.`, subject to the conditions that
1. The set is closed under the addition operator, i.e. for any $$ \vec{l}, \vec{m} \in V $$, $$ \vec{l} + \vec{m} \in V $$.
2. The addition operation is commutative, i.e. for any $$ \vec{l}, \vec{l} \in V $$, $$ \vec{l} + \vec{m} = \vec{m} + \vec{l} $$.
3. The addition operation is associative, i.e. for any $$ \vec{l}, \vec{m}, \vec{n} \in V $$, $$ (\vec{l} + \vec{m}) + \vec{n} = \vec{l} + (\vec{m} + \vec{n}) $$.
4. There exists a zero vector $$ \vec{0} \in V $$, which is the identity element of addition, i.e. for any $$ \vec{l} \in V $$, $$ \vec{l} + \vec{0} = \vec{l} $$.
5. There should exist an inverse for every element in the set, i.e. for any $$ \vec{l} \in V $$, there exists a $$ \vec{m} \in V $$ such that $$ \vec{l} + \vec{m} = \vec{0} $$.
6. The set is closed under the multiplication operation with any real-valued scalar, i.e. for any $$ \vec{l} \in V $$ and $$ r\in \mathbb{R} $$, $$ r . \vec{l} \in V $$.
7. The multiplication operation is distributive with the addition operator, i.e. for any $$ r, s \in \mathbb{R} $$ and $$ \vec{l}, \vec{m} \in V $$, $$ (r + s).\vec{l} = r.\vec{l} + s.\vec{l} $$ and $$ r.(\vec{l} + \vec{m}) = r.\vec{l} + r.\vec{m} $$.
8. The multiplication operation is compatible with the scalar multiplication operation, i.e. for any $$ r, s \in \mathbb{R} $$ and any $$ \vec{l} \in V $$, $$ r.(s.\vec{l}) = (rs).\vec{l} $$.
9. There exists an identity element of scalar multiplication, i.e. for any $$ \vec{l} \in V $$, $$ 1.\vec{l} = \vec{l} $$.


### Inner Product Space
An Inner Product Space is a vector space $$V$$ with an inner product operation, which is a mapping $$ \langle\cdot, \cdot\rangle: V \times V \to \mathbb{R} $$. Note that the range can either be $$ \mathbb{R} $$ or $$ \mathbb{C} $$, but we will only consider the case for $$ \mathbb{R} $$ here. The inner product should satisfy the following properties.
1. $$ \langle \vec{l}, \vec{m} \rangle = \langle \vec{m}, \vec{l} \rangle $$ for any $$ \vec{l}, \vec{m} \in V $$.
2. $$ \langle r.\vec{l}, \vec{m} \rangle = r\langle \vec{l}, \vec{m} \rangle $$ and $$ \langle \vec{l} + \vec{m}, \vec{n} \rangle = \langle \vec{l}, \vec{n} \rangle + \langle \vec{m}, \vec{n} \rangle $$, for any $$ r\in \mathbb{r} $$ and $$ \vec{l}, \vec{m}, \vec{n} \in V $$.
3. $$ \langle \vec{l}, \vec{l}\rangle > 0 $$ for any $$ \vec{l} \in V $$ and $$ \langle \vec{l}, \vec{l}\rangle = 0 \iff \vec{l} = \vec{0} $$.


### Scalar, Vector, Matrix and Tensor

+ A __Scalar__ is just a single number.

+ A __Vector__ is an array of numbers. A vector is an element of the inner product space $$ \mathbb{R}^n $$. For example a vector $$\mathbf{x}$$ that has $$ n $$ elements is represented as:

$$\mathbf{x}=\left[\begin{matrix}x_{1}\\ x_{2}\\ x_{3}\\ \vdots\\ x_{n}\end{matrix}\right]$$

A numpy example of a 6-element vector is given as follows:

```python
np.array([1, 2, 3, 4, 5, 6])  # a row vector that has 6 elements
```

__Technical Note__: by default, numpy can only represent row vector as numpy needs two dimensions to represent a column vector.

+ A __Matrix__ is a 2D array of numbers. Matrices are mainly used as linear operators to transform a vector space $$\mathbb{R}^m $$ to $$\mathbb{R}^n $$, which would be an $$ n\times m $$ matrix represented as:

$$\mathbf{A}=\left[\begin{matrix}A_{11} & A_{12} & \cdots & A_{1m} \\
A_{21} & A_{22} & \cdots & A_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
A_{n1} & A_{n2} & \cdots & A_{nm}\end{matrix}\right]$$

A numpy example:

```python
np.array([[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12]])  # a 3x4 matrix
```

The __matrix transpose__ is defined as $$\mathbf{A}^{\top}$$ where $$(\mathbf{A}^{\top}_{i,j})=\mathbf{A}_{j,i}$$. The transpose of the matrix can be thought of as a mirror image across the main diagonal. Python has a nice API for the matrix:

```python
A_transpose = A.T
```

+ A multi-dimensional array is called a __tensor__. Note that scalars are 0-dimensional tensors, vectors are 1-dimensional tensors and matrices are 2-dimensional tensors.

```python
np.ones(shape=(2, 3, 4, 5))  # a 4D tensor that has 2x3x4x5 elements which are filled as 1
```

__Remarks__: We do not intend to present a full review of Linear Algebra. For those who need to quickly learn the material, please read [Chapter 2 of the Deep Learning Book](http://www.deeplearningbook.org/contents/linear_algebra.html) or [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf). Both resources give a very good presentation on the topic.

## Basic Numpy
The contents of this section are mainly based on the [quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html) of Numpy from the official website.

The main object in Numpy is the homogeneous multi-dimensional array (or a tensor). The main difference between a Python multi-dimensional list and the numpy array is that elements of a list can be of different types, while the elements of a numpy array are of the same type.

Numpy's array class is called the `ndarray`, which also goes by the alias `array`. A few important attributes of the ndarray object are `ndarray.shape` which has the dimensions of the array, `ndarray.dtype` which has the type of elements in the array (e.g., numpy.int16, numpy.float16, numpy.float32, etc).

Let us look at an example.

```python
# this statement imports numpy with the alias np
# which is easier to use than the whole word 'numpy'
import numpy as np

# creates an array (0, 1, 2, ..., 14)
a = np.arange(15)

# should output (15,)
a.shape

# reshapes the above array of shape (15,) into (3, 5)
# notice the number of elements is still the same
a = a.reshape(3, 5)

# should output (3, 5)
a.shape

# type of the elements in the array
# should output int64, since it is the default dtype for the function arange
a.dtype

# recasts the elements of a into type int8
a = a.astype(np.int8)

# should output int8
a.dtype
```

To operate on numpy arrays, they have to be created. Numpy arrays can be created in many ways.

```python
# initialize from a list with the dtype float32, the dtype unless specified is int64 since
# the numbers in the list are integers
np.array([[1, 2], [3, 4]], dtype=np.float32)

# created an array initialized with zeros, the default dtype is float64
np.zeros(shape=(2, 3))

# fill an array with ones
np.ones(shape=(2, 3))

# linspace like the matlab function
np.linspace(0, 15, 30)  # 30 numbers from 0 to 15
```

Now that the arrays are created, let us look at how the arrays can be operated on. The arithmetic operations on arrays are applied element wise.

```python

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[1, 1], [2, 2]], dtype=np.float32)
# subtracting one array from another element wise
a - b

# adding two arrays element wise
a + b

# multiplying two arrays element wise
a * b

# squaring the elements of each array
a ** 2

# applying the sine function on the array multiplied with pi / 2
np.sin(a * np.pi / 2)

# for the matrix product, there is a dot function
np.dot(a, b)

# element wise exponential of the array subtracted by 2
np.exp(a - 2)

# square root of the array element wise
np.sqrt(a)
```
Arrays of different types can be operated, the resulting array corresponds to the dtype of the more general or the more precise one.
```python
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
b = np.array([[1, 1], [2, 2]], dtype=np.float32)

c = a + b

# should be of type float 64
c.dtype

a = a.astype(np.int64)
c = a + b

# should be of type float 64
c.dtype
```
Numpy also provides inplace operations to modify existing arrays.
```python
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[1, 1], [2, 2]], dtype=np.int32)

# adds the matrix b to the matrix a
a += b

# note that when trying to add a to b you get an error
b += a
```
There are many inbuilt unary operations as well, the names are self explanatory.
```python
a = np.array([[1, 2], [3, 4]])

# sum of all elements in the array
a.sum()

# sum of all elements along a particular axis
a.sum(axis=0)

# minimum of all elements in the array
a.min()

# -1 corresponds to the last dimension, -2 for the last but one and so on
# computes the cumulative sum along the last axis
a.cumsum(axis=-1)  
```
While 1D arrays can be indexed just like python native lists, multi-dimensional arrays can have one index per axis. These indices are in an n-length tuple for an n-dimensional array.
```python
a = np.arange(12)

a[2:5]  # indexes the 2nd element to the 4th element
# notice that the last element is the 4th element and not the 5th

a = a.reshape(3, 2, 2)

a[:, 1, :]  # a simple colon represents all the elements in that dimension

# 1: indexes the 1st element to the last element whole
# :-1 indexes the 0th element to the last but one element
a[1:, :, 0]
```
Iterating over multidimensional arrays is done with respect to the first axis.
```python
a = np.arange(12).reshape(3, 2, 2)

for element in a:
  print(element)

for element in np.transpose(a, axes=[1, 0, 2]):
  print(element)
```
Numpy broadcasts arrays of different shapes during arithmetic operations. Broadcasting allows functions to deal with inputs that do not have the same shape but expects inputs that have the same shape.

The first rule of broadcasting is that if all the input arrays do not have the same dimensions, a '1' will be **prepended** to the shapes of the smaller arrays until all the arrays have the same number of dimensions.

The second rule ensures that arrays with size '1' along a particular dimension acts as if they had size of the largest array in that dimension, with the value repeated in that dimension.

After these two rules, the arrays must be of same shape, otherwise the arrays are not broadcastable. Further details can be found [here](https://docs.scipy.org/doc/numpy-dev/user/basics.broadcasting.html).
```python
a = np.arange(4)
b = np.arange(5)

a + b  # throws an exception

c = a.reshape(4, 1) + b

c.shape
```

__Remarks__: For a more complete Python Numpy Tutorial, please check [this document](http://cs231n.github.io/python-numpy-tutorial/) from Stanford CS231n class.

### Basic Symbolic Computation

While classical computing (numerical computing) defines variables and uses operations to modify their values, symbolic computation defines a graph of operations on symbols, which can be substituted for values later.
These operations can include addition, subtraction, multiplication, matrix multiplication, differentiation, exponentiation, etc.

Every operation takes as input symbols (tensors), and output symbols that can be further operated upon.

While there are multiple open sourced symbolic computation libraries, like Tensorflow, Theano, etc, we will be using Keras, a high level API running on top of Tensorflow or Theano. We will be using the backend of Tensorflow in our sessions.

First let us import the backend functions of keras in python and implement some basic operations.

```python
import numpy as np
from keras import backend as K
```
Now initialize two input scalars (shape () tensors) which can then be added together. Placeholders are basic tensor variables which can later be substituted with numpy arrays during the evaluation of the further operations.
```python
input_1 = K.placeholder(shape=())
input_2 = K.placeholder(shape=())

print(input_1)

inputs_added = input_1 + input_2

print(inputs_added)
```

Now we can instantiate an add function from the symbols created above.

```python
add_function = K.function(inputs=[input_1, input_2],
                          outputs=(inputs_added,))
print(add_function)
```

This add function takes on two scalars as inputs and returns a scalar as an output.

```python
add_function((37, 42))
```

Similarly, you can also add two matrices of the same shape instead of scalars.

```python

input_1 = K.placeholder(shape=(2, 2))
input_2 = K.placeholder(shape=(2, 2))

inputs_added = input_1 + input_2

add_function = K.function(inputs=(input_1, input_2),
                          outputs=(inputs_added,))

add_function((np.array([[1, 3], [2, 4]]),
              np.array([[3, 2], [5, 6]])))
```

The main advantage of using symbolic computation is automatic differentiation, which is crucial in deep learning.

For this, we need to get acquainted with keras variables. While keras placeholders are a way to instantiate tensors, they are placeholder tensors for users to substitute values into to carry out their intended computation. To be able to use the automatic differentiation in keras, we need to define variables, with respect to which we can differentiate other symbols.

```python
# variable can be initialized with a value like this
init_variable_1 = np.zeros(shape=(2, 2))
variable_1 = K.variable(value=init_variable_1)

# variable can also be initialized with particular functions like this
variable_2 = K.ones(shape=(2, 2))

add_tensor = variable_1 + variable_2

print(variable_1)
print(add_tensor)

# notice the difference in the types, one is a variable tensor
# while the other is just a tensor

# we can evaluate the value of variables like this
K.eval(variable_1)
K.eval(variable_2)

# we can create the add function from the add_tensor just like before

add_function = K.function(inputs=(variable_1, variable_2),
                          outputs=(add_tensor,))

add_function((np.array([[1, 3], [2, 4]]),
              np.array([[3, 2], [5, 6]])))

# notice that the add_function created is independent of the variables
# the value of variables created is not affected

K.eval(variable_1)
K.eval(variable_2)

# we can set the value of variables like this
K.set_value(x=variable_1, value=np.array([[1, 3], [2, 4]]))

K.eval(variable_1)

# notice that the change in variable_1 is reflected when you evaluate
# add_tensor now

K.eval(add_tensor)
```

We can also compute more than one thing at the same time by using multiple outputs. Say we want to add two tensors, subtract two tensors, perform an element-wise squaring operation on one of the tensors and get the element-wise exponential of the other tensor.

```python

variable_1 = K.ones(shape=(2, 2))
variable_2 = K.ones(shape=(2, 2))

add_tensor = variable_1 + variable_2
subtract_tensor = variable_1 - variable_2
square_1_tensor = variable_1 ** 2
exp_2_tensor = K.exp(variable_2)

multiple_output_function = K.function(inputs=(variable_1, variable_2),
                                      outputs=(add_tensor,
                                               subtract_tensor,
                                               square_1_tensor,
                                               exp_2_tensor))

multiple_output_function((np.array([[1, 3], [2, 4]]),
                          np.array([[3, 2], [5, 6]])))
```

Now we can get to the important part of differentiating with respect to the variables. Once we have created the variables and performed operations of interest on them, we would like to get the gradients of the output symbols from those operations with respect to the variables.

```python
variable_1 = K.ones(shape=(2, 2))
variable_2 = K.ones(shape=(2, 2))

square_1_tensor = variable_1 ** 2
exp_tensors_added = K.exp(variable_1) + K.exp(variable_2)

# we can compute the gradients with respect to a single variable or
# a list of variables

# computing the gradient of square_1_tensor with respect to variable_1
grad_1_tensor = K.gradients(loss=square_1_tensor, variables=[variable_1])

# computing the gradient of square_1_tensor with respect to variable_2
grad_2_tensor = K.gradients(loss=square_1_tensor, variables=[variable_2])

# computing the gradient of exp_tensors_added with respect to both
# variable_1 and variable_2
grad_3_tensor = K.gradients(loss=exp_tensors_added,
                            variables=[variable_1,
                                       variable_2])

# we can now create functions corresponding to these operations
grad_functions = K.function(inputs=(variable_1, variable_2),
                             outputs=(grad_1_tensor[0],
                                      grad_3_tensor[0],
                                      grad_3_tensor[1]))

grad_functions((np.array([[1, 3], [2, 4]]),
                np.array([[3, 2], [5, 6]])))
```

__Remarks__: The complete API reference is available at [Keras documentation for backend](https://keras.io/backend/).

### Exercises

1. Create three symbolic placeholder vectors of length 5 (shape (5,) tensors) $$\mathbf{a}$$, $$\mathbf{b}$$ and $$\mathbf{c}$$; then create a function to compute the expression $$\mathbf{a}^2 + \mathbf{b}^2 + \mathbf{c}^2 + 2\mathbf{b}\mathbf{c}$$.

1. Create a scalar variable $$\mathbf{x}$$ and compute the tanh function on $$\mathbf{x}$$ using the exponential function. Then compute the derivative of the tanh with respect to $$\mathbf{x}$$ using the gradients function. Invoke the functions with the values -100, -1, 0, 1 and 100 to analyze the function and its derivative.

1. Create shape (2,) variable $$\mathbf{w}$$ and the shape (1,) variable $$\mathbf{b}$$. Create shape (2,) placeholder $$\mathbf{x}$$. Now create the function corresponding to $$f(w_1 * x_1 + w_2 * x_2 + b)$$ where $$f(x) = \frac{1}{1+e^{-x}}$$ and compute the gradient with respect to $$\mathbf{w}$$. Analyse the implemented operation. Then see how the function and the gradient behave for different values of the variables and the placeholder.

1. For an arbitrary $$n$$, create an $$n$$-degree polynomial for an input scalar variable $$\mathbf{x}$$ with $$(n+1)$$ variables and compute the gradients of the polynomial with respect to each of the variables.

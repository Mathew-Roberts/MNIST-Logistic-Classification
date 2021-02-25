{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\\rightarrow$Run All).\n",
    "\n",
    "Make sure you fill in any place that says `YOUR CODE HERE` and that you delete any **raise NotImplementedError()** once you have filled in your code. Enter your student identifier below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "STUDENT ID = \"200878566\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "fdd840b9ab44a389a644d4ff41790a9a",
     "grade": false,
     "grade_id": "cell-6ded9e10fb35754f",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "# MTH786U/P final assessment template\n",
    "\n",
    "This is the coding template for the final assessment of MTH786U/P in 2020/2021.\n",
    "\n",
    "The goal of this assessment is to classify hand-written digits from the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) and to present your results in a written report (at most 8 pages). The assessment is formed of three parts: 1) filling in the missing parts of this Jupyter notebook, 2) applying learned concepts from this notebook and the module MTH786 in general to the MNIST classification problem, and 3) presenting your results in a written report (written in $\\LaTeX$). \n",
    "\n",
    "Author: [Martin Benning](mailto:m.benning@qmul.ac.uk)\n",
    "\n",
    "Date: 18.11.2020\n",
    "\n",
    "Follow the instructions in this template in order to complete the first part of your assessment. Please only modify cells where you are instructed to do so. Failure to comply may result in unexpected errors that can lead to mark deductions. We load the Numpy and Matplotlib libraries. Please do not add any additional libraries here but at a later stage if required."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "5bd7074d1695e70e8c0a936b29704da2",
     "grade": false,
     "grade_id": "cell-a633ccf06b277795",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.style.use('dark_background')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "8b8af802f1a0231178e470649ea44639",
     "grade": false,
     "grade_id": "cell-1521ab82136c8e27",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "## Binary logistic regression\n",
    "\n",
    "For the first part of your final assessment you are required to implement the logistic regression model for binary classification problems as introduced in the lectures. Following up on what you have learned in the lectures and tutorials, complete the following tasks.\n",
    "\n",
    "Write a function ***logistic_function*** that takes an argument named _inputs_ and returns the output of the logistic function, i.e.\n",
    "\n",
    "\\begin{align*}\n",
    "\\frac{1}{1 + \\exp(-x)} \\, ,\n",
    "\\end{align*}\n",
    "\n",
    "applied to the input. Here $x$ is the mathematical notation for the argument _inputs_."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "f7a31ab463bfd733a5550c57433d06c3",
     "grade": false,
     "grade_id": "logistic-function",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def logistic_function(inputs):\n",
    "    # YOUR CODE HERE\n",
    "    value = 1/(1+np.exp(-(inputs)))\n",
    "    return value"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "9d4000621fded3ac2c0b5634489af1d6",
     "grade": false,
     "grade_id": "cell-002e14c765e68b53",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Test your function with the following unit tests. Passing this test will be awarded with **2 marks**. Please note that not all unit tests are visible to you."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "ade774ac5cf141a9edbb4f46f707c70d",
     "grade": true,
     "grade_id": "logistic-function-test",
     "locked": true,
     "points": 2,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "from numpy.testing import assert_array_almost_equal, assert_array_equal\n",
    "test_inputs = np.array([[0], [np.log(5)], [-3], [np.log(3)], [1]])\n",
    "assert_array_almost_equal(logistic_function(test_inputs), np.array([[1/2], [5/6], [0.04742587317756], \\\n",
    "                            [3/4], [np.exp(1)/(1 + np.exp(1))]]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "04ff095166cb1042179a534a57f7f5c4",
     "grade": false,
     "grade_id": "cell-8e2b052a692ae3ca",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "For the next exercise, write two functions that implement the objective function for binary logistic regression as well as its gradient, as defined in the lecture notes. The function for the objective function is named **binary_logistic_regression_cost_function** and should take the NumPy arrays _data_matrix_, _weights_ and _outputs_ as arguments. Here, _data_matrix_ is supposed to be a polynomial basis matrix, while _weights_ denotes the vector of weight parameters and _outputs_ is the vector of binary outputs (with values in $\\{0, 1\\}$). In order to generate a polynomial basis matrix, fill in the function **polynomial_basis**. You can follow the [solution](https://qmplus.qmul.ac.uk/mod/resource/view.php?id=1416413) of [Coursework 4](https://qmplus.qmul.ac.uk/pluginfile.php/2220881/mod_resource/content/4/coursework04.pdf) or use your own version, as long as it is consistent with the function header specified in the next cell and with the requested output. Subsequently, write a method **binary_logistic_regression_gradient** that takes the same inputs as **binary_logistic_regression_cost_function** and computes the gradient of the binary logistic regression cost function as defined in the lecture."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "32f2b5001d7a1881ffa12451d5133cb8",
     "grade": false,
     "grade_id": "logistic-cost-function",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def polynomial_basis(inputs, degree=1):\n",
    "    \n",
    "    basis_matrix = np.ones((len(inputs), 1))\n",
    "    for counter in range(1, degree + 1):\n",
    "        basis_matrix = np.c_[basis_matrix, np.power(inputs, counter)]\n",
    "    return basis_matrix\n",
    "   \n",
    "\n",
    "def binary_logistic_regression_cost_function(data_matrix, weights, outputs):\n",
    "    s = data_matrix.shape[0]\n",
    "    values = []\n",
    "    for i in range(s):\n",
    "        result = np.log(1 + np.exp(data_matrix[i,:]@weights)) - (outputs[i]*(data_matrix[i,:]@weights))\n",
    "        values.append(result)\n",
    "    return np.sum(values)\n",
    "    \n",
    "    \n",
    "def binary_logistic_regression_gradient(data_matrix, weights, outputs):\n",
    "    result = data_matrix.T @ (logistic_function(data_matrix@weights) - outputs)\n",
    "    return result\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "766d8951af9708ef7c8c6be80d721575",
     "grade": false,
     "grade_id": "cell-ee5c7e3b09c4f3f4",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "After writing Python functions for the binary logistic regression cost function and its gradient, fill in the following notebook functions for the implementation of a gradient descent method. For the first function it is acceptable to follow the solution of Coursework 6, or to use your own version if is consistent with function header and output. For the second gradient descent function named **gradient_descent_v2**, modify the gradient descent method to include a stopping criterion that ensures that gradient descent stops once\n",
    "\n",
    "\\begin{align*}\n",
    "\\| \\nabla L(w^k) \\| \\leq \\text{tolerance}\n",
    "\\end{align*}\n",
    "\n",
    "is satisfied. Here $L$ and $w^k$ are the mathematical representations of the objective _objective_ and the weight vector _weights_, at iteration $k$. The parameter _tolerance_ is a non-negative threshold that controls the Euclidean norm of the gradient. The function **gradient_descent_v2** takes the arguments _objective_, _gradient_, _initial_weights_, _step_size_, _no_of_iterations_, _print_output_ and _tolerance_. The arguments _objective_ and _gradient_ are functions that can take (weight-)arrays as arguments and return the scalar value of the objective, respectively the array representation of the corresponding gradient. The argument _initial_weights_ specifies the initial value of the variable over which you iterate. The argument _step_size_ is the gradient descent step-size parameter, the argument _no_of_iterations_ specifies the maximum number of iterations, _print_output_ determines after how many iterations the function produces a text output and _tolerance_ controls the norm of the gradient as described in the equation above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "0a0df2f9342e9204a48960b5885f38d6",
     "grade": false,
     "grade_id": "gradient_descent",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def gradient_descent(objective, gradient, initial_weights, step_size=1, no_of_iterations=100, print_output=10):\n",
    "    objective_values = []\n",
    "    weights = np.copy(initial_weights)\n",
    "    objective_values.append(objective(weights))\n",
    "    \n",
    "    for counter in range(no_of_iterations):\n",
    "        weights -= step_size * gradient(weights)\n",
    "        objective_values.append(objective(weights))\n",
    "        \n",
    "        if (counter + 1) % print_output == 0:\n",
    "            print(\"Iteration {k}/{m}, objective = {o}.\".format(k=counter+1, \\\n",
    "                    m=no_of_iterations, o=objective_values[counter]))\n",
    "            \n",
    "    print(\"Iteration completed after {k}/{m}, objective = {o}.\".format(k=counter + 1, \\\n",
    "                m=no_of_iterations, o=objective_values[counter]))\n",
    "    \n",
    "    return weights, objective_values\n",
    "\n",
    "    \n",
    "def gradient_descent_v2(objective, gradient, initial_weights, step_size=1, no_of_iterations=100, \\\n",
    "                        print_output=10, tolerance=1e-6):\n",
    "    \n",
    "    #This creates values for the cost function for each weight\n",
    "    objective_values = []\n",
    "    weights = np.copy(initial_weights)\n",
    "    objective_values.append(objective(weights))\n",
    "    iteration = 0\n",
    "    #This enforces a condition on the one norm of the gradient of the loss function\n",
    "    while np.sum(np.abs(gradient(weights))) < tolerance:\n",
    "        weights -= step_size * gradient(weights)\n",
    "        objective_values.append(objective(weights))\n",
    "        \n",
    "        iteration +=1\n",
    "        if (iteration + 1) % print_output == 0:\n",
    "            print(\"Iteration {k}, objective = {o}.\".format(k=iteration+1, \\\n",
    "                     o=objective_values[iteration]))\n",
    "            \n",
    "    print(\"Iteration completed after {k}/{m}, objective = {o}.\".format(k=iteration + 1, \\\n",
    "                 o=objective_values[iteration]))\n",
    "    \n",
    "    return weights, objective_values\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "90e8d7a7ec95458b09266c3d92aaaec6",
     "grade": false,
     "grade_id": "cell-c3a6dd3a899dcb5c",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "In the following cell, write a function **standardise** that standardises the columns of a two-dimensional NumPy array _data_matrix_."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "924547dca886c6e5532970ba42bdb82c",
     "grade": false,
     "grade_id": "standardise",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def standardise(data_matrix):\n",
    "    row_of_means = np.mean(data_matrix,axis = 0)\n",
    "    standardised_matrix = data_matrix - row_of_means\n",
    "    row_of_stds = np.std(standardised_matrix, axis = 0)\n",
    "  \n",
    "    return (standardised_matrix/row_of_stds)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "33f3d6517199ac0dd790047f327a3631",
     "grade": false,
     "grade_id": "cell-f74f777a7e81f822",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Test your results with the following cell. A total of **3 marks** will be awarded if your function passes the following standard tests. Please note that not all tests are visible to you."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "a0ea50e31317e7e314d2a7d90e46ff41",
     "grade": true,
     "grade_id": "standardisation-test",
     "locked": true,
     "points": 3,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "test_matrix = np.array([[1, 2], [3, 4], [5, 6]])\n",
    "assert_array_almost_equal(standardise(test_matrix), np.array([[-1.22474487, -1.22474487], \\\n",
    "                            [0, 0],[1.22474487, 1.22474487]]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "faa82c0c746059f5e089fd2a8bc80f46",
     "grade": false,
     "grade_id": "cell-4ec03cc4fda49034",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "To train a simple binary classifier, you require some data. The following cell calls a function that allows you to load the height-weight-gender dataset that you already know from your coursework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "40ef881fba829b6964f8b1d65fcb6a47",
     "grade": false,
     "grade_id": "cell-d3175e8ccd63fab4",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from data_loader import load_data\n",
    "height, weight, biological_sex = load_data()\n",
    "biological_sex = np.double(biological_sex).reshape(-1, 1)\n",
    "plt.plot(height, weight, '.', linewidth=3)\n",
    "plt.xlabel('Height in metres', fontsize=16)\n",
    "plt.xticks(fontsize=16)\n",
    "plt.ylabel('Weight in kilogram', fontsize=16)\n",
    "plt.yticks(fontsize=16)\n",
    "plt.tight_layout;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "93ddcbc533969ca2585a24d87b6f10ff",
     "grade": false,
     "grade_id": "cell-1d8cb3da4dac45fa",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "In the following cell, write code that initialises a polynomial data matrix _data_matrix_ of degree one with the standardised inputs formed from the height and weight arrays of the dataset. Define an objective function _objective_ with argument _weights_ based on the **binary_logistic_regression_cost_function** with fixed arguments _data_matrix_ and _biological_sex_. Repeat the same exercise to create a function _gradient_ based on **binary_logistic_regression_gradient**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "c7d63d2842004f750b77a6454bf5006d",
     "grade": false,
     "grade_id": "cell-65673f36b848eab4",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1.          1.94406149  2.50579697]\n",
      " [ 1.          0.62753668  0.02710064]\n",
      " [ 1.          2.01244346  1.59780623]\n",
      " ...\n",
      " [ 1.         -0.64968792 -1.02672965]\n",
      " [ 1.          0.69312469  0.07512745]\n",
      " [ 1.         -1.14970831 -1.48850724]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "#creating the polynomial basis array\n",
    "height_standardised = standardise(height)\n",
    "weight_standardised = standardise(weight)\n",
    "weight_height_std = np.c_[height_standardised,weight_standardised]\n",
    "data_matrix = polynomial_basis(weight_height_std)\n",
    "#checking the polynomial array\n",
    "print(data_matrix)\n",
    "\n",
    "#creating the cost function and gradient\n",
    "objective = lambda weights: binary_logistic_regression_cost_function(data_matrix, weights, biological_sex)\n",
    "gradient = lambda weights: binary_logistic_regression_gradient(data_matrix, weights, biological_sex)\n",
    "\n",
    "\n",
    " "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "726095861ab7131322e489c8a33d15f5",
     "grade": false,
     "grade_id": "cell-5c25ca5d2e19856b",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Call gradient descent with the following cell to compute _optimal_weights_ for your model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "7753f6ed0ac1f5a4adbb1ef268caae81",
     "grade": false,
     "grade_id": "cell-fb31b6af5bbbcc9a",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 1000/7000, objective = 2091.9568591203943.\n",
      "Iteration 2000/7000, objective = 2091.30072155257.\n",
      "Iteration 3000/7000, objective = 2091.297983425523.\n",
      "Iteration 4000/7000, objective = 2091.2979712556394.\n",
      "Iteration 5000/7000, objective = 2091.2979712013266.\n",
      "Iteration 6000/7000, objective = 2091.297971201084.\n",
      "Iteration 7000/7000, objective = 2091.297971201083.\n",
      "Iteration completed after 7000/7000, objective = 2091.297971201083.\n"
     ]
    }
   ],
   "source": [
    "initial_weights = np.zeros((data_matrix.shape[1], 1))\n",
    "optimal_weights, objective_values = gradient_descent(objective, gradient, initial_weights, \\\n",
    "                                    step_size=1.9/(np.linalg.norm(data_matrix, 2) ** 2), \\\n",
    "                                    no_of_iterations=7000, print_output=1000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "c84469bd2889feaf8a713f0dfb3d716a",
     "grade": false,
     "grade_id": "cell-da55ede3853660cf",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "A correct result of your gradient-descent-based logistic regression strategy will be awarded **4 marks**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "0102eed55ffb0f81e5ba3f42cffd9422",
     "grade": true,
     "grade_id": "cell-8090fbc611c0a194",
     "locked": true,
     "points": 4,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The optimal weights are w = [[-0.01870311  1.89527459 -6.3680829 ]].T with objective value L(w) = 2091.297971201083.\n"
     ]
    }
   ],
   "source": [
    "print(\"The optimal weights are w = {w}.T with objective value L(w) = {o}.\".format(w = optimal_weights.T, \\\n",
    "        o=objective_values[-1]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "14f46ce484ad38b6e849796544e9ad6c",
     "grade": false,
     "grade_id": "cell-18f1b30c908aaae8",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Write two functions **prediction_function** and **classification_accuracy** that turn your predicitons into classification results and that compare how many labels have been classified correctly. The function **prediction_function** takes the arguments _data_matrix_ and _weights_ as inputs and returns a vector of class labels with binary values in $\\{0, 1\\}$ as its output. The function **classification_accuracy** takes two inputs _true_labels_ and _recovered_labels_ and returns the percentage of correctly classified labels divided by 100."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "dda3261c6408de6bfac72ee8a2e3d197",
     "grade": false,
     "grade_id": "prediction-and-accuracy",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def prediction_function(data_matrix, weights):\n",
    "    results = logistic_function(data_matrix @ weights)\n",
    "    binary = np.where(results >0.5,1,0)\n",
    "    return binary\n",
    "   \n",
    "def classification_accuracy(true_labels, recovered_labels):\n",
    "    equal_labels = (recovered_labels == true_labels)\n",
    "    return np.mean(equal_labels)\n",
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "cd533efa2d5716dd04fe9c649cb9f0bf",
     "grade": false,
     "grade_id": "cell-747f695d7dde1390",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "The correct classification accuracy is awarded **4 marks**. The total marks possible in this section are **13 marks**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "977c57bb1188016adfa151474f8baecf",
     "grade": true,
     "grade_id": "accuracy-test",
     "locked": true,
     "points": 4,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The classification accuracy for the training set is 91.94 %.\n"
     ]
    }
   ],
   "source": [
    "print(\"The classification accuracy for the training set is {p} %.\".format(p = 100 * \\\n",
    "        classification_accuracy(biological_sex, prediction_function(data_matrix, optimal_weights))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "00633505cdd1f8712777a761fbe95961",
     "grade": false,
     "grade_id": "cell-66ec567aec881880",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "## Multinomial logistic regression\n",
    "\n",
    "This concludes the binary classification part of the first part of the final assessment. We now move on to multinomial logistic regression for multi-class classfication problems. As a first exercise, implement the softmax function **softmax_function** as defined in the lectures. The function takes the NumPy array _argument_ as its main argument, but also has an optional _axis_ argument to determine across which array-dimension you apply the softmax operation. If this argument is not specified (or set to _None_), then the softmax operation is applied to the entire array. Make sure your function works at least for NumPy arrays _argument_ with arbitrary numerical values and dimension one or two."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "4977cb40c6c166eebb31fe6ef824918a",
     "grade": false,
     "grade_id": "softmax",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.76528029 0.71807976]\n",
      " [0.23049799 0.01775346]\n",
      " [0.00422172 0.26416678]]\n"
     ]
    }
   ],
   "source": [
    "def softmax_function(argument, axis=None):\n",
    "    \n",
    "    if argument.ndim == 1:\n",
    "        new_array = np.exp(argument)\n",
    "        result = new_array/np.sum(new_array)\n",
    "        \n",
    "        return result\n",
    "    elif argument.ndim > 1:\n",
    "        \n",
    "        if axis == None:\n",
    "            result = np.exp(argument)/sum(sum(np.exp(x)) for x in argument)\n",
    "            \n",
    "            return result\n",
    "        \n",
    "        elif axis == 0:\n",
    "            new_array = np.exp(argument)\n",
    "            result = new_array/new_array.sum(axis=0, keepdims = True)\n",
    "            \n",
    "            return result\n",
    "                \n",
    "        elif axis == 1:\n",
    "            new_array = np.exp(argument)\n",
    "            result = new_array/new_array.sum(axis=1, keepdims = True)\n",
    "          \n",
    "            return result\n",
    "    \n",
    "print(softmax_function(np.array([[1.5,3], [0.3,-0.7], [-3.7,2]]),axis=0))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "d93ee498f6cc2c5167fb18f05b55cb72",
     "grade": false,
     "grade_id": "cell-0bd9e5f6e5931b4e",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Test your softmax function with the following cell. Passing this test is awarded with **4 marks**. Please note that, as usual, some tests are hidden."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "92dadfc8cfb35300c37378165909b4a0",
     "grade": true,
     "grade_id": "cell-b7751e0278da3533",
     "locked": true,
     "points": 4,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The softmax of [[ 1.5  0.3 -3.7]].T is [[0.76528029 0.23049799 0.00422172]].T.\n"
     ]
    }
   ],
   "source": [
    "argument = np.array([[1.5], [0.3], [-3.7]])\n",
    "print(\"The softmax of {arg}.T is {out}.T.\".format(arg=argument.T, out=softmax_function(argument).T))\n",
    "assert_array_almost_equal(softmax_function(np.array([[1.5], [0.3], [-3.7]])), np.array([[0.76528029], \\\n",
    "                                                        [0.23049799], [0.00422172]]))\n",
    "assert_array_almost_equal(softmax_function(np.array([[1.5, 3], [0.3, -0.7], [-3.7, 2]]), axis=0), \\\n",
    "                          np.array([[0.76528029, 0.71807976], [0.23049799, 0.01775346], \\\n",
    "                                    [0.00422172, 0.26416678]]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "22c6fcff95effedaa57877d34de89392",
     "grade": false,
     "grade_id": "cell-941464e24613e0a9",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Next, write a function **one_hot_vector_encoding** that converts an NumPy array _labels_ with values in the range of $\\{0, K - 1\\}$ into so-called one-hot vector encodings. For example, for $K = 3$ and a label vector $\\text{labels} = \\left( \\begin{matrix} 2 & 0 & 1 & 2\\end{matrix} \\right)^\\top$, the output of **one_hot_vector_encoding(labels)** should be a two-dimensional NumPy array of the form\n",
    "\n",
    "\\begin{align*}\n",
    "\\left( \\begin{matrix} 0 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\end{matrix} \\right) \\, . \n",
    "\\end{align*}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "94bcefd8f6f2834778c5e510f9797c83",
     "grade": false,
     "grade_id": "one-hot-vector-encoding",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def one_hot_vector_encoding(labels):\n",
    "    a = labels\n",
    "    b = np.zeros((a.size, a.max()+1))\n",
    "    b[np.arange(a.size),a] = 1\n",
    "    return b   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "740154e524df161eed6ffcb3b6daf01f",
     "grade": false,
     "grade_id": "cell-4d72a85842f6c692",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Implement the cost function and gradient for the multinomial logistic regression in terms of two functions **multinomial_logistic_regression_cost_function** and **multinomial_logistic_regression_gradient**. As in the binary classification case, the arguments are the polynomial data matrix _data_matrix_ and weights that are now named _weight_matrix_. Instead of passing on labels as _outputs_ as in the binary case, you pass the one hot vector encoding representation _one_hot_vector_encodings_ as your third argument. Return the cost function value, respectively the gradient, following the mathematical formulas in the lecture notes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "6866c7079139fd8a29d4c148a6e2f034",
     "grade": false,
     "grade_id": "multinomial_logistic_regression",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def multinomial_logistic_regression_cost_function(data_matrix, weight_matrix, one_hot_vector_encodings):\n",
    "    #loss function\n",
    "    s = data_matrix.shape[0]\n",
    "    K = one_hot_vector_encodings.shape[1]\n",
    "    values = 0\n",
    "    \n",
    "    for i in range(s):\n",
    "        term1s = []\n",
    "        col = np.where(one_hot_vector_encodings[i,:] == 1)[0][0]\n",
    "        term2 = (np.dot(data_matrix[i,:],weight_matrix[:,col]))\n",
    "        for j in range(K): \n",
    "            term1_before = (np.exp(np.dot(data_matrix[i,:],weight_matrix[:,j])))                  \n",
    "            term1s.append(term1_before)\n",
    "        term1 = np.log((np.sum(term1s)))\n",
    "        values += term1 - term2\n",
    "    return values\n",
    "    \n",
    "def multinomial_logistic_regression_gradient(data_matrix, weight_matrix, one_hot_vector_encodings):\n",
    "    result = data_matrix.T @ (softmax_function(data_matrix@weight_matrix,axis = 1) - one_hot_vector_encodings)\n",
    "    return result"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "a034361a28a9e882f7d7ad4328070a9c",
     "grade": false,
     "grade_id": "cell-b810132e15131dc9",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Test your implementation on the [UCI wine dataset](https://archive.ics.uci.edu/ml/datasets/wine); the dataset contains 13 attributes from a chemical analysis of Italian wines from three different cultivars. For more information on the dataset visit [this link](https://archive.ics.uci.edu/ml/datasets/wine). The code in the following cell loads the dataset and stores the labels in a NumPy array _labels_ and the attributes in a NumPy array _inputs_."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "c66a8b68f9903af0351c13f30605934f",
     "grade": false,
     "grade_id": "cell-d080a23a2048227b",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "wines = np.loadtxt('wine.data', delimiter=',')\n",
    "labels = wines[:, 0].astype(int) - 1\n",
    "inputs = wines[:, 1::]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "6a9db194ed9b4c4b458c753d58b0196e",
     "grade": false,
     "grade_id": "cell-d26a44b054d22e4d",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Transform the labels _labels_ into a one hot vector representation with your function **one_hot_vector_encoding** and store your results in a NumPy array named _outputs_."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "9378cf561d0799b518319252eaeb6c87",
     "grade": false,
     "grade_id": "cell-7362e79b5354d63b",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# YOUR CODE HERE\n",
    "outputs = one_hot_vector_encoding(labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "a365aa01296ecd58f3550a5a83dc403d",
     "grade": false,
     "grade_id": "cell-100a0ea7469b3821",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "In the following cell, write code that initialises a polynomial data matrix _data_matrix_ of degree one with the standardised inputs _inputs_ from the wine dataset. Define an objective function _objective_ with argument _weight_matrix_ based on the **multinomial_logistic_regression_cost_function** with fixed arguments _data_matrix_ and _one_hot_vector_encodings_. Repeat the same exercise to create a function _gradient_ based on **multinomial_logistic_regression_gradient**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "712836b79b8e6389ce84a45f8439a300",
     "grade": false,
     "grade_id": "cell-094a5bddd94a009a",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# YOUR CODE HERE\n",
    "\n",
    "data_matrix = polynomial_basis(standardise(inputs))\n",
    "\n",
    "objective = lambda weight_matrix: multinomial_logistic_regression_cost_function(data_matrix, weight_matrix, outputs) \n",
    "gradient = lambda weight_matrix: multinomial_logistic_regression_gradient(data_matrix, weight_matrix, outputs)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "d06a10bf709742803be0723fd183611d",
     "grade": false,
     "grade_id": "cell-901f918df8fdef00",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Call gradient descent with the following cell to compute an _optimal_weight_matrix_ for your model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "acf5d4074e6b0b65252ef485d45d5838",
     "grade": false,
     "grade_id": "cell-175babea73e2db36",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 5000/100000, objective = 0.37363264592740286.\n",
      "Iteration 10000/100000, objective = 0.19859836381135887.\n",
      "Iteration 15000/100000, objective = 0.13605627122241737.\n",
      "Iteration 20000/100000, objective = 0.10372479744703211.\n",
      "Iteration 25000/100000, objective = 0.08391801536513466.\n",
      "Iteration 30000/100000, objective = 0.07051872085948618.\n",
      "Iteration 35000/100000, objective = 0.06084138658523397.\n",
      "Iteration 40000/100000, objective = 0.05351961411115225.\n",
      "Iteration 45000/100000, objective = 0.047784005082136094.\n",
      "Iteration 50000/100000, objective = 0.04316789618475969.\n",
      "Iteration 55000/100000, objective = 0.03937164510108371.\n",
      "Iteration 60000/100000, objective = 0.036193967173282715.\n",
      "Iteration 65000/100000, objective = 0.0334945959405184.\n",
      "Iteration 70000/100000, objective = 0.031172765703073146.\n",
      "Iteration 75000/100000, objective = 0.02915420629986798.\n",
      "Iteration 80000/100000, objective = 0.027382962414542966.\n",
      "Iteration 85000/100000, objective = 0.02581606984776652.\n",
      "Iteration 90000/100000, objective = 0.024419988124662062.\n",
      "Iteration 95000/100000, objective = 0.02316814830816938.\n",
      "Iteration 100000/100000, objective = 0.022039229212158062.\n",
      "Iteration completed after 100000/100000, objective = 0.022039229212158062.\n"
     ]
    }
   ],
   "source": [
    "initial_weight_matrix = np.zeros((data_matrix.shape[1], outputs.shape[1]))\n",
    "optimal_weight_matrix, objective_values = gradient_descent(objective, gradient, initial_weight_matrix, \\\n",
    "                                    step_size=1.9/(np.linalg.norm(data_matrix, 2) ** 2), \\\n",
    "                                    no_of_iterations=100000, print_output=5000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "e5a8d433e17eb215eab8b141e3fa48bd",
     "grade": false,
     "grade_id": "cell-b7bf0daf68390bac",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Write a function **multinomial_prediction_function** that turns your predicitons into labels. The function takes the arguments _data_matrix_ and _weight_matrix_ as inputs and returns a vector of labels with values in $\\{0, K - 1 \\}$ as its output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "e824634474ea1ee537d80abc1bd092cd",
     "grade": false,
     "grade_id": "cell-2da6bb5a7e17ea5e",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def multinomial_prediction_function(data_matrix, weight_matrix):\n",
    "    k = weight_matrix.shape[1]\n",
    "    matrix = logistic_function(data_matrix@weight_matrix)\n",
    "    ohv_matrix = (matrix == matrix.max(axis=1,keepdims = 1)).astype(float)\n",
    "    prediction = ohv_matrix @ np.array(np.arange(0,k))\n",
    "    return prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "4550fa0d0fcc9f5244cf0ca66c5bf8a5",
     "grade": false,
     "grade_id": "cell-9036bda8b3b0ced4",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "The correct classification accuracy is awarded **4 marks**. The total number of possible marks in this section is **8 marks**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "bbf3d6b935f497fab4bd93eeea119319",
     "grade": true,
     "grade_id": "cell-49fe4e66273baf80",
     "locked": true,
     "points": 4,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The classification accuracy for the wine training set is 100.0 %.\n"
     ]
    }
   ],
   "source": [
    "print(\"The classification accuracy for the wine training set is {p} %.\".format(p = 100 * \\\n",
    "        classification_accuracy(labels, multinomial_prediction_function(data_matrix, optimal_weight_matrix))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "ba0705811c857f5094055fbd9f385917",
     "grade": false,
     "grade_id": "cell-45dcd721e2869a9a",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "## Ridge logistic regression\n",
    "\n",
    "For the next part, modify the multinomial logistic regression problem to include a squared Frobenius norm of the weights as a regularisation term, similar to ridge regression where we added a multiple of the squared Euclidean norm of the weights to the mean squared error. Write two functions **ridge_logistic_regression_cost_function** and **ridge_logistic_regression_gradient** that take the arguments _data_matrix_, _weight_matrix_, _one_hot_vector_encodings_ and _regularisation_parameter_ as inputs. The function **ridge_logistic_regression_cost_function** returns the evulation of the multinomial logistic regression cost function with its linear model being determined by the polynomial basis matrix _data_matrix_ and the weight matrix _weight_matrix_, plus _regularisation_parameter_ times the squared Frobenius norm of _weight_matrix_ divided by two. The function **ridge_logistic_regression_gradient** is supposed to compute the corresponding gradient."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "acdbb33c938f33ee566b9dee3262d846",
     "grade": false,
     "grade_id": "cell-abadca310df6699f",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def ridge_logistic_regression_cost_function(data_matrix, weight_matrix, one_hot_vector_encodings, \\\n",
    "                                            regularisation_parameter):\n",
    "    ridge_term = (regularisation_parameter/2) * (np.linalg.norm(weight_matrix)**2)\n",
    "    return multinomial_logistic_regression_cost_function(data_matrix, weight_matrix, one_hot_vector_encodings) + ridge_term\n",
    "    \n",
    "    \n",
    "    \n",
    "def ridge_logistic_regression_gradient(data_matrix, weight_matrix, one_hot_vector_encodings, \\\n",
    "                                       regularisation_parameter):\n",
    "    ridge_term = regularisation_parameter * weight_matrix\n",
    "    return multinomial_logistic_regression_gradient(data_matrix,weight_matrix,one_hot_vector_encodings) + ridge_term \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "c0996e4e5192b31e3a4934a13a883b09",
     "grade": false,
     "grade_id": "cell-1af3c4a5f531e2fa",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Set your regularisation parameter _regularisation_parameter_ to the value 15 and define an objective function _objective_ as well as a gradient function _gradient_, both with argument _weight_matrix_, for fixed _data_matrix_ and _outputs_ as from the wine dataset that you have used before."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "f40622d6042d8793ce03206d450ee648",
     "grade": false,
     "grade_id": "cell-3f1796626df02a88",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# YOUR CODE HERE\n",
    "regularisation_parameter = 15\n",
    "objective = lambda weight_matrix: ridge_logistic_regression_cost_function(data_matrix, weight_matrix, outputs,regularisation_parameter) \n",
    "gradient = lambda weight_matrix: ridge_logistic_regression_gradient(data_matrix, weight_matrix, outputs,regularisation_parameter)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "8d13bd2ced951479730e1afad72283eb",
     "grade": false,
     "grade_id": "cell-a0c298f30e585bd6",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Test your solution with the following cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "4d5eec1441b9057a16855a9953bfe215",
     "grade": false,
     "grade_id": "cell-4ac1dd08e2ba7c80",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 10/100, objective = 50.66582189248284.\n",
      "Iteration 20/100, objective = 48.11155251656332.\n",
      "Iteration 30/100, objective = 47.821933497251294.\n",
      "Iteration 40/100, objective = 47.766837844563554.\n",
      "Iteration 50/100, objective = 47.7524214796791.\n",
      "Iteration 60/100, objective = 47.747879295725696.\n",
      "Iteration 70/100, objective = 47.7462923106324.\n",
      "Iteration 80/100, objective = 47.74570110157326.\n",
      "Iteration 90/100, objective = 47.745470493345195.\n",
      "Iteration 100/100, objective = 47.745377257110924.\n",
      "Iteration completed after 100/100, objective = 47.745377257110924.\n"
     ]
    }
   ],
   "source": [
    "initial_weight_matrix = np.zeros((data_matrix.shape[1], outputs.shape[1]))\n",
    "ridge_weight_matrix, ridge_objective_values = gradient_descent(objective, gradient, initial_weight_matrix, \\\n",
    "                                    step_size=1.9/np.linalg.norm(data_matrix.T @ data_matrix + \\\n",
    "                                    regularisation_parameter * np.eye(data_matrix.shape[1]), 2), \\\n",
    "                                    no_of_iterations=100, print_output=10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "2920f8564006d977e1d1f3cc19c46989",
     "grade": false,
     "grade_id": "cell-ad40701d626892d4",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "The correct classification accuracy is awarded **4 marks**, which is also the total number of possible points in this section."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "a093e26a90163695684c1f98e8cfd645",
     "grade": true,
     "grade_id": "cell-3d84066ed6f97945",
     "locked": true,
     "points": 4,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The ridge regression classification accuracy with regularisation parameter 15 for the wine training set is 99.43820224719101 %.\n"
     ]
    }
   ],
   "source": [
    "print(\"The ridge regression classification accuracy with regularisation parameter {a}\".format(a = \\\n",
    "       regularisation_parameter), \"for the wine training set is {p} %.\".format(p = 100 * \\\n",
    "        classification_accuracy(labels, multinomial_prediction_function(data_matrix, ridge_weight_matrix))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "3e59581e2eed10dab4a2d0ca63e5e67d",
     "grade": false,
     "grade_id": "cell-681ea7611e1b8d47",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "## LASSO-type logistic regression\n",
    "\n",
    "As a final exercise before you are left to continue with the main part of your project of classifying handwritten digits, implement a modification of the multinomial logistic regression problem that contains a positive multiple of the one-norm of the weight matrix as regularisation term, and approximate a solution numerically with the proximal gradient descent method as introduced in the lectures. You can make use of computational solutions for coursework 9.\n",
    "\n",
    "Begin by completing the following three functions. The function **soft_thresholding** takes the two arguments _argument_ and _threshold_ and returns the solution of the soft-thresholding operation applied to _argument_ with threshold _threshold_. \n",
    "\n",
    "The function **lasso_logistic_regression_cost_function** is supposed to implement the multinomial logistic regression loss with additional one-norm regularisation for polynomial basis matrix _data_matrix_, weight matrix _weight_matrix_, the one hot vector encoding output _one_hot_vector_encodings_ and the regularisation parameter _regularisation_parameter_.\n",
    "\n",
    "The function **proximal_gradient_descent** takes the same arguments as the function **gradient_descent**, with additional argument _proximal_map_ in order to specify the proximal map to be used. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "bdab43cdab7db63c950b7d0a6c06e564",
     "grade": false,
     "grade_id": "cell-b2608195afc72125",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "def soft_thresholding(argument, threshold):\n",
    "    return np.sign(argument) * np.maximum(0, np.abs(argument) - threshold)\n",
    "    \n",
    "def lasso_logistic_regression_cost_function(data_matrix, weight_matrix, one_hot_vector_encodings, \\\n",
    "                                       regularisation_parameter):\n",
    "    lasso_term = regularisation_parameter * np.linalg.norm(weight_matrix,1)\n",
    "    return multinomial_logistic_regression_cost_function(data_matrix, weight_matrix, one_hot_vector_encodings) + lasso_term\n",
    "    \n",
    "def proximal_gradient_descent(objective, gradient, proximal_map, initial_weights, step_size=1, \\\n",
    "                              no_of_iterations=100, print_output=100):\n",
    "    objective_values = []\n",
    "    weights = initial_weights    \n",
    "    objective_values.append(objective(weights))\n",
    "    for counter in range(no_of_iterations):\n",
    "        weights = proximal_map(weights - step_size * gradient(weights))\n",
    "        objective_values.append(objective(weights))\n",
    "        if (counter + 1) % print_output == 0:\n",
    "            print(\"Iteration {k}/{m}, objective = {o}.\".format(k=counter+1, \\\n",
    "                    m=no_of_iterations, o=objective_values[counter]))\n",
    "    print(\"Iteration completed after {k}/{m}, objective = {o}.\".format(k=counter + 1, \\\n",
    "                m=no_of_iterations, o=objective_values[counter]))\n",
    "    return weights, objective_values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "2bfbe1872c304938510aeafc7fa10c25",
     "grade": false,
     "grade_id": "cell-e3ac148e879c476f",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "In the next cell, define a suitable objective function _objective_ and a suitable gradient function _gradient_, both with argument _weight_matrix_ and fixed _data_matrix_ and _outputs_ from the wine dataset that you have used before, as well as a suitable proximal map function _proximal_map_ with correctly chosen threshold."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "5af740fe918364e9026fad9423073f3d",
     "grade": false,
     "grade_id": "cell-3afeb025acbc0916",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "regularisation_parameter = 0.5\n",
    "step_size = 1.9/(np.linalg.norm(data_matrix, 2) ** 2)\n",
    "# YOUR CODE HERE\n",
    "\n",
    "objective = lambda weight_matrix: lasso_logistic_regression_cost_function(data_matrix, weight_matrix, outputs, \\\n",
    "                                       regularisation_parameter)\n",
    "gradient = lambda weight_matrix: multinomial_logistic_regression_gradient(data_matrix, weight_matrix, outputs)\n",
    "\n",
    "proximal_map = lambda weights: soft_thresholding(weights, regularisation_parameter * \\\n",
    "                                                 step_size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "8da58316078974a165e9e3fd00369e62",
     "grade": false,
     "grade_id": "cell-59e703fdde6e42f7",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Test your solution with the following cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "998771495b551e63b22c8b1980b179c9",
     "grade": false,
     "grade_id": "cell-33117d62e5699048",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 1000/10000, objective = 8.24959786201634.\n",
      "Iteration 2000/10000, objective = 8.263886291899661.\n",
      "Iteration 3000/10000, objective = 8.320757757764321.\n",
      "Iteration 4000/10000, objective = 8.354235362086085.\n",
      "Iteration 5000/10000, objective = 8.374118326110752.\n",
      "Iteration 6000/10000, objective = 8.385347232389174.\n",
      "Iteration 7000/10000, objective = 8.392015832216455.\n",
      "Iteration 8000/10000, objective = 8.396214878066097.\n",
      "Iteration 9000/10000, objective = 8.399020949261885.\n",
      "Iteration 10000/10000, objective = 8.40100378273745.\n",
      "Iteration completed after 10000/10000, objective = 8.40100378273745.\n"
     ]
    }
   ],
   "source": [
    "initial_weight_matrix = np.zeros((data_matrix.shape[1], outputs.shape[1]))\n",
    "lasso_weight_matrix, lasso_objective_values = proximal_gradient_descent(objective, gradient, proximal_map, \\\n",
    "                                                initial_weight_matrix, step_size, no_of_iterations=10000, \\\n",
    "                                                print_output=1000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "8e56343b5f34fafa2ed80f9b572b528f",
     "grade": false,
     "grade_id": "cell-69a32c9f26d37f47",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "The correct classification accuracy is awarded **5 marks**, which is also the total number of points available in this section."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "d0f049b162367cd9d765fb16be5a693e",
     "grade": true,
     "grade_id": "cell-4cd3a07610421486",
     "locked": true,
     "points": 5,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The lasso classification accuracy with regularisation parameter 0.5 for the wine training set is 100.0 %.\n"
     ]
    }
   ],
   "source": [
    "print(\"The lasso classification accuracy with regularisation parameter {a}\".format(a = \\\n",
    "       regularisation_parameter), \"for the wine training set is {p} %.\".format(p = 100 * \\\n",
    "        classification_accuracy(labels, multinomial_prediction_function(data_matrix, lasso_weight_matrix))))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "f6ffccf3240508413da555b379ba46e8",
     "grade": false,
     "grade_id": "cell-cf4145537166f52b",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "## MNIST free-style\n",
    "\n",
    "This completes the first part of the assessment. From now on, you can use all concepts that you have learned throughout this module in order to obtain a classifier that can determine which digit is seen in a $28 \\times 28$ pixel picture. While you are allowed to use different libraries for visualisation purposes, you can only use NumPy to program your classifier; tools from libraries such as SciPy are not allowed. Please note that your classfier will be tested and compared with classifiers from other students. Experimenting with differnt model approaches, regularisation models & parameters and hyperparameter-tuning strategies such cross validation is therefore highly recommended.\n",
    "\n",
    "We begin by loading the MNIST training set that is taken from this [source](http://yann.lecun.com/exdb/mnist/):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "6f85db745293111420d8bafb5f6dcc87",
     "grade": false,
     "grade_id": "cell-7b42fe502f206f1c",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The MNIST training set contains 60000 images with 784 pixels each.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "dark"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from data_loader import load_mnist\n",
    "images, labels = load_mnist('')\n",
    "print(\"The MNIST training set contains {s} images with {p} pixels each.\".format(s = images.shape[0], \\\n",
    "        p = images.shape[1]))\n",
    "plt.imshow(images[13,:].reshape(28, 28))\n",
    "plt.title(\"This is the 13th image of the MNIST training set. The corresponding label is {l}\".format( \\\n",
    "            l=labels[13]))\n",
    "plt.tight_layout;"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The cell below is a copy of the gradient descent formula, however I have removed the objective values from the return command"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradient_descent_mnist(objective, gradient, initial_weights, step_size=1, no_of_iterations=100, print_output=100):\n",
    "    objective_values = []\n",
    "    weights = np.copy(initial_weights)\n",
    "    objective_values.append(objective(weights))\n",
    "    \n",
    "    for counter in range(no_of_iterations):\n",
    "        weights -= step_size * gradient(weights)\n",
    "        objective_values.append(objective(weights))\n",
    "        \n",
    "        if (counter + 1) % print_output == 0:\n",
    "            print(\"Iteration {k}/{m}, objective = {o}.\".format(k=counter+1, \\\n",
    "                    m=no_of_iterations, o=objective_values[counter]))\n",
    "            \n",
    "    print(\"Iteration completed after {k}/{m}, objective = {o}.\".format(k=counter + 1, \\\n",
    "                m=no_of_iterations, o=objective_values[counter]))\n",
    "    \n",
    "    return weights"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "2531ffc9734a56b76172c7b9cb54c990",
     "grade": false,
     "grade_id": "cell-60965515b3dfb67b",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Use the following space to write your codes. It should be possible to reproduce results that are shown in your report with the codes that are described here. You can outsource functions into separate files if you find that this tidies up your notebook. Any additional libraries that you want to use (e.g. for visualisations etc.) can be loaded here."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "ee2bf797183742fb3d2ae82b79a82e18",
     "grade": false,
     "grade_id": "cell-a57ec7bc4606613e",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Your codes here. Please feel free to create more cells etc."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### REMEMBER\n",
    "s = 60000\n",
    "j = 784 + 1\n",
    "k = 10\n",
    "\n",
    "data_matrix = s x j\n",
    "\n",
    "weight_matrix =  j x k\n",
    "\n",
    "labels = s x k"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preparing the Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The below cells will standardise the images matrix."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def standardise_mnist(data_matrix):\n",
    "    datamatrix_copy = np.copy(data_matrix)\n",
    "    \n",
    "    row_of_means = np.mean(datamatrix_copy,axis = 0)\n",
    "    standardised_matrix = datamatrix_copy - row_of_means\n",
    "   \n",
    "    Array = np.copy(standardised_matrix)\n",
    "    \n",
    "    for col in range(datamatrix_copy.shape[1]):\n",
    "        if np.std(standardised_matrix[:,col]) == 0:\n",
    "            Array[:,col] = standardised_matrix[:,col]/1\n",
    "        else:\n",
    "            Array[:,col] = standardised_matrix[:,col]/\\\n",
    "                                                    np.std(standardised_matrix[:,col])\n",
    "    \n",
    "  \n",
    "    return Array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#-------------------------- CREATING NEW VARIABLES FROM THE DATA -----------------------------------------\n",
    "\n",
    "mnist_inputs = standardise_mnist(images)\n",
    "mnist_outputs = labels\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Partitioning (Training/Test/Validation)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This creates a partition of the data. 20% will be used for validation, whilst the other 80% will be used for training and testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Data_Partition(dataset,labels,split = 0.20):\n",
    "    dataset_copy = list(dataset)\n",
    "    labels_copy = list(labels)\n",
    "    val_data = []\n",
    "    val_labels = []\n",
    "    validation_set_size = split * len(dataset_copy)\n",
    "    while len(val_data) < validation_set_size:\n",
    "        index = np.random.randint(len(dataset_copy))\n",
    "        val_data.append(dataset_copy.pop(index))\n",
    "        val_labels.append(labels_copy.pop(index))\n",
    "    train_test_data = dataset_copy\n",
    "    train_test_labels = labels_copy\n",
    "    \n",
    "    return np.array(val_data), np.array(val_labels), np.array(train_test_data), np.array(train_test_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following cell shows the splitting of the validation and training/testing sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_data, val_labels, train_test_data, train_test_labels = Data_Partition(mnist_inputs, mnist_outputs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below is the data for the Validation Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_data,val_labels\n",
    "\n",
    "#--------------------------------- CREATING POLYNOMIAL BASIS AND ONE HOT ENCODINGS MATRICES -------------\n",
    "\n",
    "val_data_matrix = polynomial_basis(val_data,degree = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Storing the values because I don't want to continue using randomised data during the iterations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'val_data' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store val_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'val_labels' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store val_labels"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training/Testing data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below we are further spliting up the test_train set, into separate testing and training arrays. These will be done randomly, but all models will be trained on the same test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_test_data, train_test_labels # These are training/testing set before partition (48,000 samples)\n",
    "\n",
    "#--------------------------------- CREATING POLYNOMIAL BASIS AND ONE HOT ENCODINGS MATRICES -------------\n",
    "\n",
    "train_test_data_matrix = polynomial_basis(train_test_data, degree = 1)\n",
    "train_test_ohv = one_hot_vector_encoding(train_test_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_test_partition(inputdata,outputdata, prop = 0.7):\n",
    "    train_data = []\n",
    "    train_labels = []\n",
    "\n",
    "    no_training_data = prop * len(inputdata)\n",
    "    inputs = list(inputdata)\n",
    "    outputs = list(outputdata)\n",
    "    while len(train_data) < no_training_data:\n",
    "        indx = np.random.randint(len(inputs))\n",
    "        train_labels.append(outputs.pop(indx))\n",
    "        train_data.append(inputs.pop(indx))\n",
    "    test_data = inputs\n",
    "    test_labels = outputs\n",
    "    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below are the splits for the training/testing partition created by the data_partition function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data, train_labels, test_data, test_labels = train_test_partition(train_test_data_matrix,train_test_ohv) #Split sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'train_data' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store train_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'train_labels' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store train_labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'test_data' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store test_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'test_labels' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store test_labels"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#  MODELS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model A: Ridge Logistic Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following code block only uses the training and testing data sets as defined above"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_validation_ridge(regularisation_parameters,traindata = train_data, trainlabels = train_labels,\\\n",
    "                           testdata = test_data,testlabels = test_labels, iterations = 1000):\n",
    "\n",
    "    \n",
    "#----------------------------- TEST RESULTS FOR EACH PARAMETER ---------------------------------    \n",
    "    \n",
    "    \n",
    "    parameters = []\n",
    "    classification_accuracies = []\n",
    "    optimal_weight_matrices = []\n",
    "      \n",
    "    \n",
    "#---------------------------- TRAINING EACH MODEL WITH THE TRAINING SET -------------------------------    \n",
    "    \n",
    "    \n",
    "    for alpha in regularisation_parameters:\n",
    "        parameters.append(alpha)\n",
    "        \n",
    "        initial_weight_matrix = np.zeros((traindata.shape[1], trainlabels.shape[1]))\n",
    "\n",
    "        objective = lambda weight_matrix: ridge_logistic_regression_cost_function(traindata,\\\n",
    "                                                                        weight_matrix, trainlabels,alpha) \n",
    "        gradient = lambda weight_matrix: ridge_logistic_regression_gradient(traindata, \\\n",
    "                                                                            weight_matrix, trainlabels,alpha)\n",
    "        \n",
    "        optimal_weight_matrix = gradient_descent_mnist(objective, gradient, initial_weight_matrix, \\\n",
    "                                          step_size=1.9/np.linalg.norm(traindata.T @ traindata + \\\n",
    "                                            alpha * np.eye(traindata.shape[1]), 2), \\\n",
    "                                                no_of_iterations= iterations, print_output=10)\n",
    "        \n",
    "        optimal_weight_matrices.append(optimal_weight_matrix)\n",
    "        \n",
    "        \n",
    "#---------------------------- TESTING THE MODEL ON TEST LABELS -------------------------------\n",
    "        \n",
    "    \n",
    "        classifier = classification_accuracy(testlabels@np.arange(0,testlabels.shape[1]), \\\n",
    "                            multinomial_prediction_function(testdata, optimal_weight_matrix))\n",
    "            \n",
    "        classification_accuracies.append(classifier)\n",
    "        \n",
    "    return list(zip(parameters,classification_accuracies)), optimal_weight_matrices[np.argmax(classification_accuracies)]\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing 3 alpha parameters, 1,10 and 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 10/500, objective = 31373.83287106924.\n",
      "Iteration 20/500, objective = 22885.43367523952.\n",
      "Iteration 30/500, objective = 19442.286545863844.\n",
      "Iteration 40/500, objective = 17517.56167407681.\n",
      "Iteration 50/500, objective = 16264.004278608469.\n",
      "Iteration 60/500, objective = 15370.146946898285.\n",
      "Iteration 70/500, objective = 14693.45680186876.\n",
      "Iteration 80/500, objective = 14158.990240055671.\n",
      "Iteration 90/500, objective = 13723.334293645576.\n",
      "Iteration 100/500, objective = 13359.475727956336.\n",
      "Iteration 110/500, objective = 13049.654369797829.\n",
      "Iteration 120/500, objective = 12781.672024085592.\n",
      "Iteration 130/500, objective = 12546.847716262611.\n",
      "Iteration 140/500, objective = 12338.819016643203.\n",
      "Iteration 150/500, objective = 12152.805775465662.\n",
      "Iteration 160/500, objective = 11985.139819367314.\n",
      "Iteration 170/500, objective = 11832.954353307869.\n",
      "Iteration 180/500, objective = 11693.97286114733.\n",
      "Iteration 190/500, objective = 11566.362000839847.\n",
      "Iteration 200/500, objective = 11448.626819003945.\n",
      "Iteration 210/500, objective = 11339.5346441608.\n",
      "Iteration 220/500, objective = 11238.058841067397.\n",
      "Iteration 230/500, objective = 11143.336589108407.\n",
      "Iteration 240/500, objective = 11054.6367380189.\n",
      "Iteration 250/500, objective = 10971.335021385776.\n",
      "Iteration 260/500, objective = 10892.894721873068.\n",
      "Iteration 270/500, objective = 10818.851431389083.\n",
      "Iteration 280/500, objective = 10748.800926090882.\n",
      "Iteration 290/500, objective = 10682.389437824433.\n",
      "Iteration 300/500, objective = 10619.305787597154.\n",
      "Iteration 310/500, objective = 10559.274977804931.\n",
      "Iteration 320/500, objective = 10502.052935071977.\n",
      "Iteration 330/500, objective = 10447.422166120325.\n",
      "Iteration 340/500, objective = 10395.188142549987.\n",
      "Iteration 350/500, objective = 10345.176271520691.\n",
      "Iteration 360/500, objective = 10297.229341070053.\n",
      "Iteration 370/500, objective = 10251.205353206757.\n",
      "Iteration 380/500, objective = 10206.975676527087.\n",
      "Iteration 390/500, objective = 10164.42346422321.\n",
      "Iteration 400/500, objective = 10123.442294038792.\n",
      "Iteration 410/500, objective = 10083.934994902933.\n",
      "Iteration 420/500, objective = 10045.812631278739.\n",
      "Iteration 430/500, objective = 10008.993621226076.\n",
      "Iteration 440/500, objective = 9973.402968149729.\n",
      "Iteration 450/500, objective = 9938.97158943808.\n",
      "Iteration 460/500, objective = 9905.635727866058.\n",
      "Iteration 470/500, objective = 9873.336433868157.\n",
      "Iteration 480/500, objective = 9842.019108664006.\n",
      "Iteration 490/500, objective = 9811.633099797355.\n",
      "Iteration 500/500, objective = 9782.131341975046.\n",
      "Iteration completed after 500/500, objective = 9782.131341975046.\n",
      "Iteration 10/500, objective = 31377.43374741898.\n",
      "Iteration 20/500, objective = 22892.46823097318.\n",
      "Iteration 30/500, objective = 19452.03740402865.\n",
      "Iteration 40/500, objective = 17529.599145972956.\n",
      "Iteration 50/500, objective = 16278.039132543841.\n",
      "Iteration 60/500, objective = 15385.969656036496.\n",
      "Iteration 70/500, objective = 14710.907598425922.\n",
      "Iteration 80/500, objective = 14177.94267437706.\n",
      "Iteration 90/500, objective = 13743.685425756024.\n",
      "Iteration 100/500, objective = 13381.139886626881.\n",
      "Iteration 110/500, objective = 13072.558977642868.\n",
      "Iteration 120/500, objective = 12805.75469054018.\n",
      "Iteration 130/500, objective = 12572.054145842116.\n",
      "Iteration 140/500, objective = 12365.101462151846.\n",
      "Iteration 150/500, objective = 12180.121867506956.\n",
      "Iteration 160/500, objective = 12013.451662984033.\n",
      "Iteration 170/500, objective = 11862.22781879565.\n",
      "Iteration 180/500, objective = 11724.177018953917.\n",
      "Iteration 190/500, objective = 11597.46866531924.\n",
      "Iteration 200/500, objective = 11480.610175744132.\n",
      "Iteration 210/500, objective = 11372.370942411844.\n",
      "Iteration 220/500, objective = 11271.726137566206.\n",
      "Iteration 230/500, objective = 11177.814532997103.\n",
      "Iteration 240/500, objective = 11089.906389022097.\n",
      "Iteration 250/500, objective = 11007.378695205602.\n",
      "Iteration 260/500, objective = 10929.695858071991.\n",
      "Iteration 270/500, objective = 10856.394479978007.\n",
      "Iteration 280/500, objective = 10787.07124972346.\n",
      "Iteration 290/500, objective = 10721.373227008442.\n",
      "Iteration 300/500, objective = 10658.98998674204.\n",
      "Iteration 310/500, objective = 10599.647220278455.\n",
      "Iteration 320/500, objective = 10543.101485776604.\n",
      "Iteration 330/500, objective = 10489.135870401124.\n",
      "Iteration 340/500, objective = 10437.556380511114.\n",
      "Iteration 350/500, objective = 10388.18891703506.\n",
      "Iteration 360/500, objective = 10340.876724920701.\n",
      "Iteration 370/500, objective = 10295.478229892811.\n",
      "Iteration 380/500, objective = 10251.865194341568.\n",
      "Iteration 390/500, objective = 10209.921138245898.\n",
      "Iteration 400/500, objective = 10169.539981722659.\n",
      "Iteration 410/500, objective = 10130.624873961115.\n",
      "Iteration 420/500, objective = 10093.087179601529.\n",
      "Iteration 430/500, objective = 10056.845598589696.\n",
      "Iteration 440/500, objective = 10021.825399502417.\n",
      "Iteration 450/500, objective = 9987.957749577428.\n",
      "Iteration 460/500, objective = 9955.17912734699.\n",
      "Iteration 470/500, objective = 9923.430806006649.\n",
      "Iteration 480/500, objective = 9892.65839752.\n",
      "Iteration 490/500, objective = 9862.811449040853.\n",
      "Iteration 500/500, objective = 9833.843084550641.\n",
      "Iteration completed after 500/500, objective = 9833.843084550641.\n",
      "Iteration 10/500, objective = 31413.414999956716.\n",
      "Iteration 20/500, objective = 22962.70130626098.\n",
      "Iteration 30/500, objective = 19549.307253487703.\n",
      "Iteration 40/500, objective = 17649.576560048477.\n",
      "Iteration 50/500, objective = 16417.805129051918.\n",
      "Iteration 60/500, objective = 15543.406211945665.\n",
      "Iteration 70/500, objective = 14884.397154239978.\n",
      "Iteration 80/500, objective = 14366.203050136224.\n",
      "Iteration 90/500, objective = 13945.671271916244.\n",
      "Iteration 100/500, objective = 13595.979941339614.\n",
      "Iteration 110/500, objective = 13299.514070592986.\n",
      "Iteration 120/500, objective = 13044.18848463755.\n",
      "Iteration 130/500, objective = 12821.412073577965.\n",
      "Iteration 140/500, objective = 12624.89514191028.\n",
      "Iteration 150/500, objective = 12449.917305616118.\n",
      "Iteration 160/500, objective = 12292.86014432677.\n",
      "Iteration 170/500, objective = 12150.898752586125.\n",
      "Iteration 180/500, objective = 12021.792235875295.\n",
      "Iteration 190/500, objective = 11903.737811652833.\n",
      "Iteration 200/500, objective = 11795.266950044926.\n",
      "Iteration 210/500, objective = 11695.169988761834.\n",
      "Iteration 220/500, objective = 11602.440457380757.\n",
      "Iteration 230/500, objective = 11516.233311573304.\n",
      "Iteration 240/500, objective = 11435.833157769765.\n",
      "Iteration 250/500, objective = 11360.629768560959.\n",
      "Iteration 260/500, objective = 11290.0989972631.\n",
      "Iteration 270/500, objective = 11223.787745371299.\n",
      "Iteration 280/500, objective = 11161.302010447007.\n",
      "Iteration 290/500, objective = 11102.297301768138.\n",
      "Iteration 300/500, objective = 11046.470893928865.\n",
      "Iteration 310/500, objective = 10993.555519119142.\n",
      "Iteration 320/500, objective = 10943.31419361186.\n",
      "Iteration 330/500, objective = 10895.535944197223.\n",
      "Iteration 340/500, objective = 10850.032253295041.\n",
      "Iteration 350/500, objective = 10806.634081990347.\n",
      "Iteration 360/500, objective = 10765.1893613675.\n",
      "Iteration 370/500, objective = 10725.560866402562.\n",
      "Iteration 380/500, objective = 10687.624404906419.\n",
      "Iteration 390/500, objective = 10651.267267884754.\n",
      "Iteration 400/500, objective = 10616.386898253546.\n",
      "Iteration 410/500, objective = 10582.889742965735.\n",
      "Iteration 420/500, objective = 10550.690259889929.\n",
      "Iteration 430/500, objective = 10519.710055754176.\n",
      "Iteration 440/500, objective = 10489.877135429142.\n",
      "Iteration 450/500, objective = 10461.12524605515.\n",
      "Iteration 460/500, objective = 10433.393302171764.\n",
      "Iteration 470/500, objective = 10406.624880214544.\n",
      "Iteration 480/500, objective = 10380.767772587493.\n",
      "Iteration 490/500, objective = 10355.773593065847.\n",
      "Iteration 500/500, objective = 10331.597426572012.\n",
      "Iteration completed after 500/500, objective = 10331.597426572012.\n"
     ]
    }
   ],
   "source": [
    "ridge_accuracy, ridge_weight_matrix = model_validation_ridge([1,10,100], \\\n",
    "                                                         iterations = 500)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "([(1, 0.9064583333333334),\n",
       "  (10, 0.9064583333333334),\n",
       "  (100, 0.9059722222222222)],\n",
       " array([[-0.06684603, -0.1622626 ,  0.05406983, ..., -0.06043682,\n",
       "          0.18508697,  0.02607455],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        ...,\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ]]))"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ridge_accuracy,ridge_weight_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'ridge_accuracy' (list)\n"
     ]
    }
   ],
   "source": [
    "%store ridge_accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'ridge_weight_matrix' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store ridge_weight_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing the model on the validation data with the best weight matrix yields an 88.13% accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9095"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classification_accuracy(val_labels, multinomial_prediction_function(val_data_matrix,ridge_weight_matrix))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The below cells are testing the accuracy of the model on the classification of individual numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "ridge_data_matrix = polynomial_basis(val_data,degree = 1)\n",
    "y_pred = multinomial_prediction_function(ridge_data_matrix, ridge_weight_matrix)\n",
    "\n",
    "index = np.argsort(val_labels)\n",
    "val_labels_sort = val_labels[index]\n",
    "y_pred_sort = y_pred[index]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[97.45547073791349,\n",
       " 96.54929577464789,\n",
       " 87.90123456790123,\n",
       " 87.23958333333334,\n",
       " 93.25744308231172,\n",
       " 82.70120259019427,\n",
       " 94.88817891373802,\n",
       " 93.33333333333333,\n",
       " 84.85639686684074,\n",
       " 89.0677966101695]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "misclassifications = []\n",
    "for i in range(10):\n",
    "    index = (val_labels_sort == i)\n",
    "    a = y_pred_sort[index]\n",
    "    b = (a == val_labels_sort[index])\n",
    "    misclassifications.append(np.mean(b) * 100)\n",
    "    \n",
    "misclassifications"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model B: Multinomial Logistic Regression (no regularisation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_validation_multinomial(traindata = train_data, trainlabels = train_labels,\\\n",
    "                           testdata = test_data,testlabels = test_labels, iterations = 1000):\n",
    "       \n",
    "    \n",
    "#---------------------------- TRAINING MODEL WITH THE TRAINING SET -------------------------------    \n",
    "    \n",
    "        \n",
    "    initial_weight_matrix = np.zeros((traindata.shape[1], trainlabels.shape[1]))\n",
    "\n",
    "    objective = lambda weight_matrix: multinomial_logistic_regression_cost_function(traindata,\\\n",
    "                                                                    weight_matrix, trainlabels) \n",
    "    gradient = lambda weight_matrix: multinomial_logistic_regression_gradient(traindata, \\\n",
    "                                                                        weight_matrix, trainlabels)\n",
    "\n",
    "    optimal_weight_matrix = gradient_descent_mnist(objective, gradient, initial_weight_matrix, \\\n",
    "                                      step_size=1.9/(np.linalg.norm(traindata, 2) ** 2), \\\n",
    "                                            no_of_iterations= iterations, print_output=10)\n",
    "    \n",
    "\n",
    "#---------------------------- TESTING THE MODEL ON TEST LABELS -------------------------------\n",
    "\n",
    "\n",
    "    classifier = classification_accuracy(testlabels@np.arange(0,testlabels.shape[1]), \\\n",
    "                        multinomial_prediction_function(testdata, optimal_weight_matrix))\n",
    "\n",
    "        \n",
    "    return classifier, optimal_weight_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 10/500, objective = 31373.43274279922.\n",
      "Iteration 20/500, objective = 22884.651931497465.\n",
      "Iteration 30/500, objective = 19441.20284857824.\n",
      "Iteration 40/500, objective = 17516.22372976163.\n",
      "Iteration 50/500, objective = 16262.444193709018.\n",
      "Iteration 60/500, objective = 15368.38797608562.\n",
      "Iteration 70/500, objective = 14691.516674196835.\n",
      "Iteration 80/500, objective = 14156.882984982389.\n",
      "Iteration 90/500, objective = 13721.071330344514.\n",
      "Iteration 100/500, objective = 13357.066558384706.\n",
      "Iteration 110/500, objective = 13047.107042301746.\n",
      "Iteration 120/500, objective = 12778.993456194252.\n",
      "Iteration 130/500, objective = 12544.043926942692.\n",
      "Iteration 140/500, objective = 12335.895298037181.\n",
      "Iteration 150/500, objective = 12149.766822842781.\n",
      "Iteration 160/500, objective = 11981.989831415085.\n",
      "Iteration 170/500, objective = 11829.697110860263.\n",
      "Iteration 180/500, objective = 11690.611789917988.\n",
      "Iteration 190/500, objective = 11562.900222068274.\n",
      "Iteration 200/500, objective = 11445.067190822547.\n",
      "Iteration 210/500, objective = 11335.87979573659.\n",
      "Iteration 220/500, objective = 11234.311201036571.\n",
      "Iteration 230/500, objective = 11139.49840944947.\n",
      "Iteration 240/500, objective = 11050.710114234429.\n",
      "Iteration 250/500, objective = 10967.321909661307.\n",
      "Iteration 260/500, objective = 10888.796953741454.\n",
      "Iteration 270/500, objective = 10814.670726316406.\n",
      "Iteration 280/500, objective = 10744.538902330929.\n",
      "Iteration 290/500, objective = 10678.047621826863.\n",
      "Iteration 300/500, objective = 10614.885622213911.\n",
      "Iteration 310/500, objective = 10554.777829495051.\n",
      "Iteration 320/500, objective = 10497.480100273306.\n",
      "Iteration 330/500, objective = 10442.774876918711.\n",
      "Iteration 340/500, objective = 10390.46757174681.\n",
      "Iteration 350/500, objective = 10340.383537182135.\n",
      "Iteration 360/500, objective = 10292.3655106161.\n",
      "Iteration 370/500, objective = 10246.27144709397.\n",
      "Iteration 380/500, objective = 10201.972671569116.\n",
      "Iteration 390/500, objective = 10159.352296586056.\n",
      "Iteration 400/500, objective = 10118.303861949995.\n",
      "Iteration 410/500, objective = 10078.730161104007.\n",
      "Iteration 420/500, objective = 10040.542225252615.\n",
      "Iteration 430/500, objective = 10003.65844122569.\n",
      "Iteration 440/500, objective = 9968.003783052125.\n",
      "Iteration 450/500, objective = 9933.509140442891.\n",
      "Iteration 460/500, objective = 9900.110730058383.\n",
      "Iteration 470/500, objective = 9867.749577659732.\n",
      "Iteration 480/500, objective = 9836.371061125888.\n",
      "Iteration 490/500, objective = 9805.92450589363.\n",
      "Iteration 500/500, objective = 9776.362825709386.\n",
      "Iteration completed after 500/500, objective = 9776.362825709386.\n"
     ]
    }
   ],
   "source": [
    "multinomial_acc, multinomial_weight_matrix = model_validation_multinomial(iterations = 500)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.9064583333333334,\n",
       " array([[-0.06686982, -0.16231489,  0.05408785, ..., -0.06045681,\n",
       "          0.18514762,  0.0260845 ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        ...,\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ],\n",
       "        [ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
       "          0.        ,  0.        ]]))"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "multinomial_acc, multinomial_weight_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'multinomial_acc' (float64)\n"
     ]
    }
   ],
   "source": [
    "%store multinomial_acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'multinomial_weight_matrix' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store multinomial_weight_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing Model on Validation Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9095"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classification_accuracy(val_labels, multinomial_prediction_function(val_data_matrix,multinomial_weight_matrix))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "multi_data_matrix = polynomial_basis(val_data,degree = 1)\n",
    "y_pred = multinomial_prediction_function(multi_data_matrix, multinomial_weight_matrix)\n",
    "\n",
    "index = np.argsort(val_labels)\n",
    "val_labels_sort = val_labels[index]\n",
    "y_pred_sort = y_pred[index]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[97.45547073791349,\n",
       " 96.54929577464789,\n",
       " 87.90123456790123,\n",
       " 87.23958333333334,\n",
       " 93.25744308231172,\n",
       " 82.70120259019427,\n",
       " 94.88817891373802,\n",
       " 93.33333333333333,\n",
       " 84.85639686684074,\n",
       " 89.0677966101695]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "misclassifications = []\n",
    "for i in range(10):\n",
    "    index = (val_labels_sort == i)\n",
    "    a = y_pred_sort[index]\n",
    "    b = (a == val_labels_sort[index])\n",
    "    misclassifications.append(np.mean(b) * 100)\n",
    "    \n",
    "misclassifications"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model C: Multinomial Logistic Regression (different degrees)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "polynomial_base = []\n",
    "deg2 = polynomial_basis(train_data[:,1:],degree = 2)\n",
    "deg4 = polynomial_basis(train_data[:,1:],degree = 4)\n",
    "deg6 = polynomial_basis(train_data[:,1:],degree = 6)\n",
    "polynomial_base.append(deg2)\n",
    "polynomial_base.append(deg4)\n",
    "polynomial_base.append(deg6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "polynomial_test_base = []\n",
    "deg_test_2 = polynomial_basis(test_data[:,1:],degree = 2)\n",
    "deg_test_4 = polynomial_basis(test_data[:,1:],degree = 4)\n",
    "deg_test_6 = polynomial_basis(test_data[:,1:],degree = 6) \n",
    "polynomial_test_base.append(deg_test_2)\n",
    "polynomial_test_base.append(deg_test_4)\n",
    "polynomial_test_base.append(deg_test_6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "degrees = list(zip(polynomial_base,polynomial_test_base))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_validation_multinomial_deg(traindata, testdata,\\\n",
    "                                     trainlabels = train_labels,\\\n",
    "                                     testlabels = test_labels,\\\n",
    "                                     iterations = 1000):\n",
    "    \n",
    " \n",
    "    \n",
    "#---------------------------- TRAINING EACH MODEL WITH THE TRAINING SET -------------------------------    \n",
    "    \n",
    "        \n",
    "    initial_weight_matrix = np.zeros((traindata.shape[1], trainlabels.shape[1]))\n",
    "\n",
    "    objective = lambda weight_matrix: multinomial_logistic_regression_cost_function(traindata,\\\n",
    "                                                                    weight_matrix, trainlabels) \n",
    "    gradient = lambda weight_matrix: multinomial_logistic_regression_gradient(traindata, \\\n",
    "                                                                        weight_matrix, trainlabels)\n",
    "\n",
    "    optimal_weight_matrix = gradient_descent_mnist(objective, gradient, initial_weight_matrix, \\\n",
    "                                      step_size=1.9/(np.linalg.norm(traindata, 2) ** 2), \\\n",
    "                                            no_of_iterations= iterations, print_output=10)\n",
    "    \n",
    "\n",
    "#---------------------------- TESTING THE MODEL ON TEST LABELS -------------------------------\n",
    "\n",
    "\n",
    "    classifier = classification_accuracy(testlabels@np.arange(0,testlabels.shape[1]), \\\n",
    "                        multinomial_prediction_function(testdata, optimal_weight_matrix))\n",
    "    \n",
    "    print(classifier)\n",
    "\n",
    "        \n",
    "    return classifier, optimal_weight_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 10/500, objective = 76920.57694209095.\n",
      "Iteration 20/500, objective = 76532.29673809849.\n",
      "Iteration 30/500, objective = 76182.15869694277.\n",
      "Iteration 40/500, objective = 75858.49403888425.\n",
      "Iteration 50/500, objective = 75555.48914257948.\n",
      "Iteration 60/500, objective = 75269.3680826331.\n",
      "Iteration 70/500, objective = 74997.4319814007.\n",
      "Iteration 80/500, objective = 74737.66137717785.\n",
      "Iteration 90/500, objective = 74488.4963926648.\n",
      "Iteration 100/500, objective = 74248.70029482001.\n",
      "Iteration 110/500, objective = 74017.27062857692.\n",
      "Iteration 120/500, objective = 73793.37950656726.\n",
      "Iteration 130/500, objective = 73576.33237794714.\n",
      "Iteration 140/500, objective = 73365.53880408913.\n",
      "Iteration 150/500, objective = 73160.49122435319.\n",
      "Iteration 160/500, objective = 72960.74916990908.\n",
      "Iteration 170/500, objective = 72765.9272833842.\n",
      "Iteration 180/500, objective = 72575.68605738482.\n",
      "Iteration 190/500, objective = 72389.72455213801.\n",
      "Iteration 200/500, objective = 72207.77457659451.\n",
      "Iteration 210/500, objective = 72029.59596962892.\n",
      "Iteration 220/500, objective = 71854.97272440247.\n",
      "Iteration 230/500, objective = 71683.70977215149.\n",
      "Iteration 240/500, objective = 71515.63029051705.\n",
      "Iteration 250/500, objective = 71350.57343453153.\n",
      "Iteration 260/500, objective = 71188.39241153069.\n",
      "Iteration 270/500, objective = 71028.95283838687.\n",
      "Iteration 280/500, objective = 70872.13133228252.\n",
      "Iteration 290/500, objective = 70717.81429613693.\n",
      "Iteration 300/500, objective = 70565.89686727438.\n",
      "Iteration 310/500, objective = 70416.28200383976.\n",
      "Iteration 320/500, objective = 70268.87968798995.\n",
      "Iteration 330/500, objective = 70123.60622857229.\n",
      "Iteration 340/500, objective = 69980.38364885352.\n",
      "Iteration 350/500, objective = 69839.13914726216.\n",
      "Iteration 360/500, objective = 69699.80462097292.\n",
      "Iteration 370/500, objective = 69562.31624380093.\n",
      "Iteration 380/500, objective = 69426.61409110918.\n",
      "Iteration 390/500, objective = 69292.64180558.\n",
      "Iteration 400/500, objective = 69160.34629857923.\n",
      "Iteration 410/500, objective = 69029.67748260644.\n",
      "Iteration 420/500, objective = 68900.58803099285.\n",
      "Iteration 430/500, objective = 68773.03316152842.\n",
      "Iteration 440/500, objective = 68646.97044116788.\n",
      "Iteration 450/500, objective = 68522.35960937369.\n",
      "Iteration 460/500, objective = 68399.16241794557.\n",
      "Iteration 470/500, objective = 68277.34248552647.\n",
      "Iteration 480/500, objective = 68156.86516514591.\n",
      "Iteration 490/500, objective = 68037.69742344222.\n",
      "Iteration 500/500, objective = 67919.80773030694.\n",
      "Iteration completed after 500/500, objective = 67919.80773030694.\n",
      "0.64625\n",
      "Iteration 10/500, objective = 77319.5026205874.\n",
      "Iteration 20/500, objective = 77308.75741293613.\n",
      "Iteration 30/500, objective = 77301.63190258502.\n",
      "Iteration 40/500, objective = 77296.02089506965.\n",
      "Iteration 50/500, objective = 77291.31856392631.\n",
      "Iteration 60/500, objective = 77287.25089467755.\n",
      "Iteration 70/500, objective = 77283.6650953178.\n",
      "Iteration 80/500, objective = 77280.46228290946.\n",
      "Iteration 90/500, objective = 77277.571759389.\n",
      "Iteration 100/500, objective = 77274.93994750718.\n",
      "Iteration 110/500, objective = 77272.52492478126.\n",
      "Iteration 120/500, objective = 77270.29328332347.\n",
      "Iteration 130/500, objective = 77268.21808300994.\n",
      "Iteration 140/500, objective = 77266.27740165616.\n",
      "Iteration 150/500, objective = 77264.45325892998.\n",
      "Iteration 160/500, objective = 77262.73079777123.\n",
      "Iteration 170/500, objective = 77261.09765343399.\n",
      "Iteration 180/500, objective = 77259.54346316749.\n",
      "Iteration 190/500, objective = 77258.05948288561.\n",
      "Iteration 200/500, objective = 77256.63828581832.\n",
      "Iteration 210/500, objective = 77255.27352429563.\n",
      "Iteration 220/500, objective = 77253.95974037323.\n",
      "Iteration 230/500, objective = 77252.69221438262.\n",
      "Iteration 240/500, objective = 77251.46684307385.\n",
      "Iteration 250/500, objective = 77250.28004096413.\n",
      "Iteration 260/500, objective = 77249.12865998394.\n",
      "Iteration 270/500, objective = 77248.00992361429.\n",
      "Iteration 280/500, objective = 77246.92137260173.\n",
      "Iteration 290/500, objective = 77245.8608199306.\n",
      "Iteration 300/500, objective = 77244.82631328741.\n",
      "Iteration 310/500, objective = 77243.81610357622.\n",
      "Iteration 320/500, objective = 77242.82861840444.\n",
      "Iteration 330/500, objective = 77241.86243961295.\n",
      "Iteration 340/500, objective = 77240.91628417074.\n",
      "Iteration 350/500, objective = 77239.98898784467.\n",
      "Iteration 360/500, objective = 77239.07949118996.\n",
      "Iteration 370/500, objective = 77238.18682747375.\n",
      "Iteration 380/500, objective = 77237.31011225091.\n",
      "Iteration 390/500, objective = 77236.44853430545.\n",
      "Iteration 400/500, objective = 77235.60134777393.\n",
      "Iteration 410/500, objective = 77234.76786526643.\n",
      "Iteration 420/500, objective = 77233.94745184618.\n",
      "Iteration 430/500, objective = 77233.13951974372.\n",
      "Iteration 440/500, objective = 77232.34352370152.\n",
      "Iteration 450/500, objective = 77231.55895687216.\n",
      "Iteration 460/500, objective = 77230.78534718264.\n",
      "Iteration 470/500, objective = 77230.0222541222.\n",
      "Iteration 480/500, objective = 77229.26926587777.\n",
      "Iteration 490/500, objective = 77228.52599679271.\n",
      "Iteration 500/500, objective = 77227.79208509797.\n",
      "Iteration completed after 500/500, objective = 77227.79208509797.\n",
      "0.49701388888888887\n",
      "Iteration 10/500, objective = 77329.32638543147.\n",
      "Iteration 20/500, objective = 77323.983126283.\n",
      "Iteration 30/500, objective = 77321.18297334778.\n",
      "Iteration 40/500, objective = 77319.0621129215.\n",
      "Iteration 50/500, objective = 77317.25980453976.\n",
      "Iteration 60/500, objective = 77315.66357223141.\n",
      "Iteration 70/500, objective = 77314.22315633082.\n",
      "Iteration 80/500, objective = 77312.90889531761.\n",
      "Iteration 90/500, objective = 77311.69997425383.\n",
      "Iteration 100/500, objective = 77310.58046062854.\n",
      "Iteration 110/500, objective = 77309.53766339517.\n",
      "Iteration 120/500, objective = 77308.56127213976.\n",
      "Iteration 130/500, objective = 77307.64280908144.\n",
      "Iteration 140/500, objective = 77306.77524009517.\n",
      "Iteration 150/500, objective = 77305.95268466625.\n",
      "Iteration 160/500, objective = 77305.17019445893.\n",
      "Iteration 170/500, objective = 77304.42358156914.\n",
      "Iteration 180/500, objective = 77303.70928340609.\n",
      "Iteration 190/500, objective = 77303.02425492596.\n",
      "Iteration 200/500, objective = 77302.36588161858.\n",
      "Iteration 210/500, objective = 77301.73190853279.\n",
      "Iteration 220/500, objective = 77301.120381932.\n",
      "Iteration 230/500, objective = 77300.52960106284.\n",
      "Iteration 240/500, objective = 77299.95807814081.\n",
      "Iteration 250/500, objective = 77299.40450507434.\n",
      "Iteration 260/500, objective = 77298.86772575851.\n",
      "Iteration 270/500, objective = 77298.34671298326.\n",
      "Iteration 280/500, objective = 77297.84054917854.\n",
      "Iteration 290/500, objective = 77297.34841035603.\n",
      "Iteration 300/500, objective = 77296.86955269666.\n",
      "Iteration 310/500, objective = 77296.4033013498.\n",
      "Iteration 320/500, objective = 77295.94904106148.\n",
      "Iteration 330/500, objective = 77295.5062083202.\n",
      "Iteration 340/500, objective = 77295.07428476779.\n",
      "Iteration 350/500, objective = 77294.65279164823.\n",
      "Iteration 360/500, objective = 77294.24128512606.\n",
      "Iteration 370/500, objective = 77293.83935232107.\n",
      "Iteration 380/500, objective = 77293.4466079353.\n",
      "Iteration 390/500, objective = 77293.06269137484.\n",
      "Iteration 400/500, objective = 77292.68726428492.\n",
      "Iteration 410/500, objective = 77292.32000842028.\n",
      "Iteration 420/500, objective = 77291.96062380563.\n",
      "Iteration 430/500, objective = 77291.6088271317.\n",
      "Iteration 440/500, objective = 77291.2643503547.\n",
      "Iteration 450/500, objective = 77290.92693946103.\n",
      "Iteration 460/500, objective = 77290.59635337707.\n",
      "Iteration 470/500, objective = 77290.27236300014.\n",
      "Iteration 480/500, objective = 77289.95475033212.\n",
      "Iteration 490/500, objective = 77289.64330770494.\n",
      "Iteration 500/500, objective = 77289.3378370773.\n",
      "Iteration completed after 500/500, objective = 77289.3378370773.\n",
      "0.3534027777777778\n"
     ]
    }
   ],
   "source": [
    "classification = []\n",
    "optimal_weight_matrix = []\n",
    "for deg in degrees:\n",
    "    classification_acc, optimal_weights = model_validation_multinomial_deg(deg[0],deg[1], iterations = 500)\n",
    "    classification.append(classification_acc)\n",
    "    optimal_weight_matrix.append(optimal_weights)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "deg_classification = classification\n",
    "deg_optimal_weights = optimal_weight_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'deg_classification' (list)\n"
     ]
    }
   ],
   "source": [
    "%store deg_classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'deg_optimal_weights' (list)\n"
     ]
    }
   ],
   "source": [
    "%store deg_optimal_weights"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing on the Validation Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12000, 1569)"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# making a new polynomial basis for degree = 2\n",
    "\n",
    "new_matrix = val_data_matrix[:,1:]\n",
    "type(new_matrix)\n",
    "basis = polynomial_basis(val_data,degree=2)\n",
    "basis.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6433333333333333"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classification_accuracy(val_labels,multinomial_prediction_function(basis,deg_optimal_weights[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "deg_data_matrix = polynomial_basis(val_data,degree = 2)\n",
    "y_pred = multinomial_prediction_function(deg_data_matrix, deg_optimal_weights[0])\n",
    "\n",
    "index = np.argsort(val_labels)\n",
    "val_labels_sort = val_labels[index]\n",
    "y_pred_sort = y_pred[index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[93.21458863443596,\n",
       " 90.63380281690141,\n",
       " 62.79835390946502,\n",
       " 70.48611111111111,\n",
       " 31.78633975481611,\n",
       " 42.183163737280296,\n",
       " 86.5814696485623,\n",
       " 88.69918699186992,\n",
       " 28.37249782419495,\n",
       " 37.20338983050848]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "misclassifications = []\n",
    "for i in range(10):\n",
    "    index = (val_labels_sort == i)\n",
    "    a = y_pred_sort[index]\n",
    "    b = (a == val_labels_sort[index])\n",
    "    misclassifications.append(np.mean(b) * 100)\n",
    "    \n",
    "misclassifications"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model D: LASSO Logistic Regression "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_validation_LASSO(regularisation_parameters,traindata = train_data, trainlabels = train_labels,\\\n",
    "                           testdata = test_data,testlabels = test_labels, iterations = 1000):\n",
    "\n",
    "    \n",
    "#----------------------------- TEST RESULTS FOR EACH PARAMETER ---------------------------------    \n",
    "    \n",
    "    \n",
    "    parameters = []\n",
    "    classification_accuracies = []\n",
    "    optimal_weight_matrices = []\n",
    "    \n",
    "    \n",
    "    \n",
    "#---------------------------- TRAINING EACH MODEL WITH THE TRAINING SET -------------------------------    \n",
    "    \n",
    "    stepsize = 1.9/(np.linalg.norm(traindata, 2) ** 2)\n",
    "    for alpha in regularisation_parameters:\n",
    "        parameters.append(alpha)\n",
    "        \n",
    "        initial_weight_matrix = np.zeros((traindata.shape[1], trainlabels.shape[1]))\n",
    "        \n",
    "        objective = lambda weight_matrix: lasso_logistic_regression_cost_function(traindata,\\\n",
    "                                                                    weight_matrix, trainlabels, \\\n",
    "                                                                                   alpha)\n",
    "        gradient = lambda weight_matrix: multinomial_logistic_regression_gradient(traindata,\\\n",
    "                                                                    weight_matrix, trainlabels)\n",
    "\n",
    "        proximal_map = lambda weights: soft_thresholding(weights, alpha * \\\n",
    "                                                 stepsize)\n",
    "\n",
    "\n",
    "        \n",
    "        optimal_weight_matrix, value = proximal_gradient_descent(objective, gradient, proximal_map, \\\n",
    "                                                                 initial_weight_matrix, \\\n",
    "                                                step_size=stepsize, \\\n",
    "                              no_of_iterations= iterations, print_output=10)\n",
    "        \n",
    "        optimal_weight_matrices.append(optimal_weight_matrix)\n",
    "        \n",
    "        \n",
    "#---------------------------- TESTING THE MODEL ON TEST LABELS -------------------------------\n",
    "        \n",
    "    \n",
    "        classifier = classification_accuracy(testlabels@np.arange(0,testlabels.shape[1]), \\\n",
    "                            multinomial_prediction_function(testdata, optimal_weight_matrix))\n",
    "        \n",
    "        print(classifier)\n",
    "            \n",
    "        classification_accuracies.append(classifier)\n",
    "        \n",
    "    return list(zip(parameters,classification_accuracies)), \\\n",
    "                                optimal_weight_matrices[np.argmax(classification_accuracies)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 10/500, objective = 31571.34703825616.\n",
      "Iteration 20/500, objective = 23102.63182116647.\n",
      "Iteration 30/500, objective = 19675.592711681875.\n",
      "Iteration 40/500, objective = 17764.224186264826.\n",
      "Iteration 50/500, objective = 16522.16380821228.\n",
      "Iteration 60/500, objective = 15638.608640999439.\n",
      "Iteration 70/500, objective = 14971.31058842001.\n",
      "Iteration 80/500, objective = 14445.50386308884.\n",
      "Iteration 90/500, objective = 14017.937821511956.\n",
      "Iteration 100/500, objective = 13661.575515851067.\n",
      "Iteration 110/500, objective = 13358.869539709096.\n",
      "Iteration 120/500, objective = 13097.643828602779.\n",
      "Iteration 130/500, objective = 12869.170689061091.\n",
      "Iteration 140/500, objective = 12667.159932659482.\n",
      "Iteration 150/500, objective = 12486.880467696781.\n",
      "Iteration 160/500, objective = 12324.701218186925.\n",
      "Iteration 170/500, objective = 12177.738116967785.\n",
      "Iteration 180/500, objective = 12043.761792539111.\n",
      "Iteration 190/500, objective = 11920.985509405815.\n",
      "Iteration 200/500, objective = 11807.94938435303.\n",
      "Iteration 210/500, objective = 11703.420780614715.\n",
      "Iteration 220/500, objective = 11606.369227334242.\n",
      "Iteration 230/500, objective = 11515.938172468748.\n",
      "Iteration 240/500, objective = 11431.375444759164.\n",
      "Iteration 250/500, objective = 11352.115684451186.\n",
      "Iteration 260/500, objective = 11277.630452043184.\n",
      "Iteration 270/500, objective = 11207.445537035184.\n",
      "Iteration 280/500, objective = 11141.158911131917.\n",
      "Iteration 290/500, objective = 11078.406253746123.\n",
      "Iteration 300/500, objective = 11018.920886575908.\n",
      "Iteration 310/500, objective = 10962.401712476258.\n",
      "Iteration 320/500, objective = 10908.617740637566.\n",
      "Iteration 330/500, objective = 10857.375008791067.\n",
      "Iteration 340/500, objective = 10808.461838879546.\n",
      "Iteration 350/500, objective = 10761.717759489318.\n",
      "Iteration 360/500, objective = 10716.978918200826.\n",
      "Iteration 370/500, objective = 10674.12261300909.\n",
      "Iteration 380/500, objective = 10633.011345471774.\n",
      "Iteration 390/500, objective = 10593.51165433884.\n",
      "Iteration 400/500, objective = 10555.527997671772.\n",
      "Iteration 410/500, objective = 10518.97155833378.\n",
      "Iteration 420/500, objective = 10483.767308597236.\n",
      "Iteration 430/500, objective = 10449.816926717445.\n",
      "Iteration 440/500, objective = 10417.05993338481.\n",
      "Iteration 450/500, objective = 10385.421454339556.\n",
      "Iteration 460/500, objective = 10354.845770555246.\n",
      "Iteration 470/500, objective = 10325.270007468262.\n",
      "Iteration 480/500, objective = 10296.638118876244.\n",
      "Iteration 490/500, objective = 10268.898771200516.\n",
      "Iteration 500/500, objective = 10242.014460547685.\n",
      "Iteration completed after 500/500, objective = 10242.014460547685.\n",
      "0.9060416666666666\n"
     ]
    }
   ],
   "source": [
    "lasso_accuracies, lasso_optimal_weight_matrix = model_validation_LASSO([1],\\\n",
    "                                                                      iterations = 500)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'lasso_accuracies' (list)\n"
     ]
    }
   ],
   "source": [
    "%store lasso_accuracies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'lasso_optimal_weight_matrix' (ndarray)\n"
     ]
    }
   ],
   "source": [
    "%store lasso_optimal_weight_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing on Validation Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9084166666666667"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classification_accuracy(val_labels,multinomial_prediction_function(val_data_matrix,lasso_optimal_weight_matrix))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing classification for each digit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso_data_matrix = polynomial_basis(val_data,degree = 1)\n",
    "y_pred = multinomial_prediction_function(lasso_data_matrix, lasso_optimal_weight_matrix)\n",
    "\n",
    "index = np.argsort(val_labels)\n",
    "val_labels_sort = val_labels[index]\n",
    "y_pred_sort = y_pred[index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[97.54028837998302,\n",
       " 96.54929577464789,\n",
       " 87.57201646090536,\n",
       " 86.89236111111111,\n",
       " 93.34500875656742,\n",
       " 82.88621646623497,\n",
       " 94.96805111821087,\n",
       " 93.33333333333333,\n",
       " 84.07310704960835,\n",
       " 88.98305084745762]"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "misclassifications = []\n",
    "for i in range(10):\n",
    "    index = (val_labels_sort == i)\n",
    "    a = y_pred_sort[index]\n",
    "    b = (a == val_labels_sort[index])\n",
    "    misclassifications.append(np.mean(b) * 100)\n",
    "    \n",
    "misclassifications"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "96d581f44d3ec35e6124a992c1c895ac",
     "grade": false,
     "grade_id": "cell-ae67818ad4f4e5d7",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "Specify the code for your best data model (linear, polynomial or else) applied to your best weights and assign the result to the output of the function **model_function**. The data model has to be created from the argument _inputs_, which is a two-dimensional array where the first dimension equals the number of samples and the second dimension the dimension of the data (784 in case of MNIST). If, for example, your model is a linear basis model applied to a weight matrix _best_weight_matrix_, then your code could look like:\n",
    "\n",
    "```\n",
    "def model_function(inputs, best_weight_matrix):\n",
    "    return polynomial_basis(inputs, 1) @ best_weight_matrix\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "deletable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "8d2e6423a8f730e83447fe6590cf5c09",
     "grade": false,
     "grade_id": "cell-d1751b92b149be1b",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "'''Specify your model function. The output has to be a two-dimensional array\n",
    "    You are allowed to include additional input arguments.'''\n",
    "def model_function(inputs):\n",
    "    standardise_inputs = standardise_mnist(inputs)\n",
    "    return polynomial_basis(standardise_inputs,degree = 1) @ multinomial_weight_matrix\n",
    "    # YOUR CODE HERE"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "87eceb699c2ee18fc35f993e4e1415d5",
     "grade": false,
     "grade_id": "cell-41aef5fe41114aae",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "The code in the next cell will then evaluate the classification performance of your best classifier when applied to hidden data that is similar (but different) to the MNIST training dataset. The best classification performance will be awarded **three extra marks** (capped at 100). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "2b8cdb2d64d0b8bc862f5d1a978d5a4c",
     "grade": false,
     "grade_id": "cell-7d9f3270c59caca4",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "### CODE HIDDEN ###"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "dbb06ce796df75a4b1f095cfaf5edd24",
     "grade": false,
     "grade_id": "cell-4f318e920d827dde",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "This completes the MTH786 coding project requirements. Please do not forget to write a detailed report on your findings (at most 8 pages) with $\\LaTeX$ and your favourite editor. If no editor is at hand, please feel free to use online editors such as [Overleaf](https://www.overleaf.com)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

[Table of Contents](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

# Installation, Python, Numpy, and filterpy

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
#format the book
%matplotlib inline
from __future__ import division, print_function
import matplotlib.pyplot as plt
import book_format
book_format.load_style()
</pre>

This book is written in IPython Notebook, a browser based interactive Python environment that mixes Python, text, and math. I choose it because of the interactive features - I found Kalman filtering nearly impossible to learn until I started working in an interactive environment. It is difficult to form an intuition of the effect of many of the parameters that you can tune until you can change them rapidly and immediately see the output. An interactive environment also allows you to play 'what if' scenarios out. "What if I set <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> to zero?" It is trivial to find out with Ipython Notebook.

Another reason I choose it is because I find that a typical textbook leaves many things opaque. For example, there might be a beautiful plot next to some pseudocode. That plot was produced by software, but software that is not available to me as a reader. I want everything that went into producing this book to be available to the reader. How do you plot a covariance ellipse? You won't know if you read most books. With IPython Notebook all you have to do is look at the source code.

Even if you choose to read the book online you will want Python and the SciPy stack installed so that you can write your own Kalman filters. There are many different ways to install these libraries, and I cannot cover them all, but I will cover a few typical scenarios.

## Installing the SciPy Stack

This book requires IPython, NumPy, SciPy, SymPy, and Matplotlib. The SciPy stack depends on third party Fortran and C code, and is *not* trivial to install from source code - even the SciPy website strongly urges using a pre-built installation, and I concur with this advice.

I use the Anaconda distribution from Continuum Analytics. This is an excellent package that combines all of the packages listed above, plus many others in one distribution. Installation is very straightforward, and it can be done alongside other Python installation you might already have on your machine. It is free to use, and Continuum Analytics will always ensure that the latest releases of all it's packages are available and built correctly for your OS. You may download it from here: http://continuum.io/downloads

I strongly recommend using the latest Python 3 version that they provide; I am developing in Python 3.x, and only sporadically test that everything works in Python 2.7. However, it is my long term goal to support 2.7, and if you wish to either not be bothered with 3.x and are willing to be occasionally frustrated with breaking check ins you may stick with Python 2.7.

I am still writing the book, so I do not know exactly what versions of each package is required. I do strongly urge you to use IPython 2.0 or later (this version number has nothing to do with Python 2 vs 3, by the way), as it provides many useful features which I will explain later.

There are other choices for installing the SciPy stack. The SciPy stack provides instructions here: http://scipy.org/install.html

### Manual Install of the SciPy stack

This really isn't easy, and again I advice you to follow the advice of the SciPy website and to use a prepackaged solution. Still, I will give you one example of an install from a fresh operating system. You will have to adapt the instructions to fit your OS and OS version. I will use xubuntu, a linux distribution, because that is what I am most familiar with. I know there are pre-built binaries for Windows (link provided on the SciPy installation web page), and I have no experience with Mac and Python.

I started by doing a fresh install of xubuntu 14.04 on a virtual machine. At that point there is a version of Python 2.7.6 and 3.4 pre-installed. As discussed above you may use either version; I will give the instructions for version 3.4 both because I prefer version 3 and because it can be slightly more tricky because 2.7 is the default Python that runs in xubuntu. Basically the difference is that you have to append a '3' at the end of commands. `python3` instead of `python`, `pip3` instead of `pip`, and so on.

First we will install pip with the following command:

    sudo apt-get install python3-pip


Now you will need to install the various packages from the Ubuntu repositories. Unfortunately they usually do not contain the latest version, so we will also install the development tools necessary to perform an upgrade, which requires compiling various modules.


    sudo apt-get install python3-numpy python3-scipy python3-matplotlib
    sudo apt_get install libblas-dev liblapack-dev gfortran python3-dev


Now you can upgrade the packages. This will take a long time as everything needs to be compiled from source. If you get an error I do not know how to help you!

    sudo pip3 install numpy --upgrade
    sudo pip3 install scipy --upgrade

Now you get to install SymPy. You can download it from github (replace version number with the most recent version):

    wget https://github.com/sympy/sympy/releases/download/sympy-0.7.6/sympy-0.7.6.tar.gz
    tar -zxvf sympy-0.7.6.tar.gz

Now go into the directory you just extracted and run setup.

    sudo python3 setup.py install


If all of this went without a hitch you should have a good install. Try the following. From the command line type `ipython3` to launch ipython, then issue the following commands. Each should run with no exceptions.

    import numpy as np
    import scipy as sp
    import sympy as sym
    np.__version__
    sp.__version__
    sum.__version__

Now let's make sure plotting works correctly.

    import matplotlib.pyplot as plt
    plt.plot([1, 4, 3])
    plt.show()

Now you get to fix IPython so ipython notebook will run. First, I had to uninstall IPython with

    sudo pip3 uninstall ipython

Then, I reinstalled it with the `[all]` option so that all the required dependencies are installed.

    sudo pip3 install "ipython3[all]"

Now test the installation by typing

    ipython notebook

If successful, it should launch a browser showing you the IPython home page.


That was not fun. It actually goes somewhat smoother in Windows, where you can download pre-built binaries for these packages; however, you are dependent on this being done for you in a timely manner. So, please follow the SciPy advice and use a pre-built distribution! I will not be supporting this manual install process.

## Installing/downloading and running the book

Okay, so now you have the SciPy stack installed, how do you download the book? This is easy as the book is in a github repository. From the command line type the following:

    git clone https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.git

Alternatively, browse to https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python and download it via your browser.

Now, from the command prompt change to the directory that was just created, and then run ipython notebook:

    cd Kalman-and-Bayesian-Filters-in-Python
    ipython notebook

A browser window should launch showing you all of the chapters in the book. Browse to the first chapter by clicking on it, then open the notebook in that subdirectory by clicking on the link.

## Using IPython Notebook

A complete tutorial on IPython Notebook is beyond the scope of this book. Many are available online. In short, Python code is placed in cells. These are prefaced with text like `In [1]:`, and the code itself is in a boxed area. If you press CTRL-ENTER while focus is inside the box the code will run and the results will be displayed below the box. Like this:

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
print(3+7.2)
</pre>

If you have this open in ipython notebook now, go ahead and modify that code by changing the expression inside the print statement and pressing CTRL+ENTER. The output should be changed to reflect what you typed in the code cell.

## SymPy

SymPy is a Python package for performing symbolic mathematics. The full scope of its abilities are beyond this book, but it can perform algebra, integrate and differentiate equations, find solutions to differential equations, and much more. For example, we use use to to compute the Jacobian of matrices and the expected value integral computations.

First, a simple example. We will import SymPy, initialize its pretty print functionality (which will print equations using LaTeX). We will then declare a symbol for Numpy to use.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import sympy
sympy.init_printing(use_latex='mathjax')

phi, x = sympy.symbols('\phi, x')
phi
</pre>

Notice how we use a latex expression for the symbol `phi`. This is not necessary, but if you do it will render as LaTeX when output. Now let's do some math. What is the derivative of <span class="math-tex" data-type="tex">\\(\sqrt{\phi}\\)</span>?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
sympy.diff('sqrt(phi)')
</pre>

We can factor equations

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
sympy.factor(phi**3 -phi**2 + phi - 1)
</pre>

and we can expand them.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
((phi+1)*(phi-4)).expand()
</pre>

You can find the value of an equation by substituting in a value for a variable

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
w =x**2 -3*x +4
w.subs(x,4)
</pre>

You can also use strings for equations that use symbols that you have not defined

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
x = sympy.expand('(t+1)*2')
x
</pre>

Now let's use SymPy to compute the Jacobian of a matrix. Let us have a function

<span class="math-tex" data-type="tex">\\(h=\sqrt{(x^2 + z^2)}\\)</span>

for which we want to find the Jacobian with respect to x, y, and z.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
x, y, z = sympy.symbols('x y z')

H = sympy.Matrix([sympy.sqrt(x**2 + z**2)])

state = sympy.Matrix([x, y, z])
H.jacobian(state)
</pre>

Now let's compute the discrete process noise matrix <span class="math-tex" data-type="tex">\\(\mathbf{Q}_k\\)</span> given the continuous process noise matrix
<span class="math-tex" data-type="tex">\\(\mathbf{Q} = \Phi_s \begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}\\)</span>

and the equation

<span class="math-tex" data-type="tex">\\(\mathbf{Q} = \int_0^{\Delta t} \Phi(t)\mathbf{Q}\Phi^T(t) dt\\)</span>

where
<span class="math-tex" data-type="tex">\\(\Phi(t) = \begin{bmatrix}1 & \Delta t & {\Delta t}^2/2 \\ 0 & 1 & \Delta t\\ 0& 0& 1\end{bmatrix}\\)</span>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
dt = sympy.symbols('\Delta{t}')
F_k = sympy.Matrix([[1, dt, dt**2/2],
                    [0,  1,      dt],
                    [0,  0,       1]])
Q = sympy.Matrix([[0,0,0],
                  [0,0,0],
                  [0,0,1]])

sympy.integrate(F_k*Q*F_k.T,(dt, 0, dt))
</pre>

[Table of Contents](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

# Preface

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
#format the book
import book_format
book_format.load_style()
</pre>

Introductory textbook for Kalman filters and Bayesian filters. All code is written in Python, and the book itself is written in IPython Notebook so that you can run and modify the code in the book in place, seeing the results inside the book. What better way to learn?

<img src="https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/animations/05_dog_track.gif">

## Reading Online

You may access this book via nbviewer at any time by using this address:
[*Read Online Now*](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

The quickest way to get starting with reading the book is to read it online using the link above. The book is written as a collection of IPython Notebooks, an interactive, browser based system that allows you to combine text, Python, and math into your browser. The website http://nbviewer.org provides an IPython Notebook server that renders notebooks stored at github (or elsewhere). The rendering is done in real time when you load the book. If you read my book today, and then I make a change tomorrow, when you go back tomorrow you will see that change. Perhaps more importantly, the book uses animations to demonstrate how the algorithms perform over time. The PDF version of the book, discussed in the next paragraph, cannot show the animations.

The preface available from the link above has all the information in this README and more, so feel free to follow the link now.

I periodically generate a PDF of the book from the Notebooks. I do not do this for every check in, so the PDF will usually lag the content in github and on nbviewer.org. However, I do generate it whenever I make a substantial change. Of course, you will not be able to run and modify the code in the notebooks, nor will you be able to see the animations.

## PDF Version

I periodically generate a PDF of the book from the Notebooks. I do not do this for every check in, so the PDF will usually lag the content in github and on nbviewer.org. However, I do generate it whenever I make a substantial change. Of course, you will not be able to run and modify the code in the notebooks, nor will you be able to see the animations.

[*PDF Version of the book*](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Kalman_and_Bayesian_Filters_in_Python.pdf)

##Downloading the book

However, this book is intended to be interactive and I recommend using it in that form. If you install IPython on your computer and then clone this book you will be able to run all of the code in the book yourself. You can perform experiments, see how filters react to different data, see how different filters react to the same data, and so on. I find this sort of immediate feedback both vital and invigorating. You do not have to wonder "what happens if". Try it and see!

The github pages for this project are at

    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

You can clone it to your hard drive with the command

`git clone https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.git`

This will create a directory named `Kalman-and-Bayesian-Filters-in-Python`. Navigate to the directory, and run IPython notebook with the command

    ipython notebook

This will open a browser window showing the contents of the base directory. The book is organized into chapters. To read Chapter 2, click on the link for chapter 2. This will cause the browser to open that subdirectory. In each subdirectory there will be one or more IPython Notebooks (all notebooks have a .ipynb file extension). The chapter contents are in the notebook with the same name as the chapter name. There are sometimes supporting notebooks for doing things like generating animations that are displayed in the chapter. These are not intended to be read by the end user, but of course if you are curious as to how an animation is made go ahead and take a look.

This is admittedly a somewhat cumbersome interface to a book; I am following in the footsteps of several other projects that are somewhat re-purposing IPython Notebook to generate entire books. I feel the slight annoyances have a huge payoff - instead of having to download a separate code base and run it in an IDE while you try to read a book, all of the code and text is in one place. If you want to alter the code, you may do so and immediately see the effects of your change. If you find a bug, you can make a fix, and push it back to my repository so that everyone in the world benefits. And, of course, you will never encounter a problem I face all the time with traditional books - the book and the code are out of sync with each other, and you are left scratching your head as to which source to trust.

## Version 0.0

Not ready for public consumption. In development.

##Motivation

This is a book for programmers that have a need or interest in Kalman filtering. The motivation for this book came out of my desire for a gentle introduction to Kalman filtering. I'm a software engineer that spent almost two decades in the avionics field, and so I have always been 'bumping elbows' with the Kalman filter, but never implemented one myself. They always has a fearsome reputation for difficulty, and I did not have the requisite education. Everyone I met that did implement them had multiple graduate courses on the topic and extensive industrial experience with them. As I moved into solving tracking problems with computer vision the need to implement them myself became urgent. There are classic textbooks in the field, such as Grewal and Andrew's excellent *Kalman Filtering*. But sitting down and trying to read many of these books is a dismal and trying experience if you do not have the background. Typically the first few chapters fly through several years of undergraduate math, blithely referring you to textbooks on, for example, Itō calculus, and presenting an entire semester's worth of statistics in a few brief paragraphs. These books are good textbooks for an upper undergraduate course, and an invaluable reference to researchers and professionals, but the going is truly difficult for the more casual reader. Symbology is introduced without explanation, different texts use different words and variables names for the same concept, and the books are almost devoid of examples or worked problems. I often found myself able to parse the words and comprehend the mathematics of a definition, but had no idea as to what real world phenomena these words and math were attempting to describe. "But what does that *mean?*" was my repeated thought.

However, as I began to finally understand the Kalman filter I realized the underlying concepts are quite straightforward. A few simple probability rules, some intuition about how we integrate disparate knowledge to explain events in our everyday life and the core concepts of the Kalman filter are accessible. Kalman filters have a reputation for difficulty, but shorn of much of the formal terminology the beauty of the subject and of their math became clear to me, and I fell in love with the topic.

As I began to understand the math and theory more difficulties itself. A book or paper's author makes some statement of fact and presents a graph as proof.  Unfortunately, why the statement is true is not clear to me, nor is the method by which you might make that plot obvious. Or maybe I wonder "is this true if R=0?"  Or the author provides pseudocode - at such a high level that the implementation is not obvious. Some books offer Matlab code, but I do not have a license to that expensive package. Finally, many books end each chapter with many useful exercises. Exercises which you need to understand if you want to implement Kalman filters for yourself, but exercises with no answers. If you are using the book in a classroom, perhaps this is okay, but it is terrible for the independent reader. I loathe that an author withholds information from me, presumably to avoid 'cheating' by the student in the classroom.

None of this necessary, from my point of view. Certainly if you are designing a Kalman filter for a aircraft or missile you must thoroughly master of all of the mathematics and topics in a typical Kalman filter textbook. I just want to track an image on a screen, or write some code for my Arduino project. I want to know how the plots in the book are made, and chose different parameters than the author chose. I want to run simulations. I want to inject more noise in the signal and see how a filter performs. There are thousands of opportunities for using Kalman filters in everyday code, and yet this fairly straightforward topic is the provenance of rocket scientists and academics.

I wrote this book to address all of those needs. This is not the book for you if you program avionics for Boeing or design radars for Raytheon. Go get a degree at Georgia Tech, UW, or the like, because you'll need it. This book is for the hobbyist, the curious, and the working engineer that needs to filter or smooth data.

This book is interactive. While you can read it online as static content, I urge you to use it as intended. It is written using IPython Notebook, which allows me to combine text, python, and python output in one place. Every plot, every piece of data in this book is generated from Python that is available to you right inside the notebook. Want to double the value of a parameter? Click on the Python cell, change the parameter's value, and click 'Run'. A new plot or printed output will appear in the book.

This book has exercises, but it also has the answers. I trust you. If you just need an answer, go ahead and read the answer. If you want to internalize this knowledge, try to implement the exercise before you read the answer.

This book has supporting libraries for computing statistics, plotting various things related to filters, and for the various filters that we cover. This does require a strong caveat; most of the code is written for didactic purposes. It is rare that I chose the most efficient solution (which often obscures the intent of the code), and in the first parts of the book I did not concern myself with numerical stability. This is important to understand - Kalman filters in aircraft are carefully designed and implemented to be numerically stable; the naive implementation is not stable in many cases. If you are serious about Kalman filters this book will not be the last book you need. My intention is to introduce you to the concepts and mathematics, and to get you to the point where the textbooks are approachable.

Finally, this book is free. The cost for the books required to learn Kalman filtering is somewhat prohibitive even for a Silicon Valley engineer like myself; I cannot believe the are within the reach of someone in a depressed economy, or a financially struggling student. I have gained so much from free software like Python, and free books like those from Allen B. Downey [here](http://www.greenteapress.com/) [1]. It's time to repay that. So, the book is free, it is hosted on free servers, and it uses only free and open software such as IPython and mathjax to create the book.

##Installation and Software Requirements

** author's note**. *The book is still being written, and so I am not focusing on issues like supporting multiple versions of Python. I am staying more or less on the bleeding edge of Python 3 for the time being. If you follow my suggestion of installing Anaconda all of the versioning problems will be taken care of for you, and you will not alter or affect any existing installation of Python on your machine. I am aware that telling somebody to install a specific packaging system is not a long term solution, but I can either focus on endless regression testing for every minor code change, or work on delivering the book, and then doing one sweep through it to maximize compatibility. I opt for the latter. In the meantime I welcome bug reports if the book does not work on your platform.*

If you want to run the notebook on your computer, which is what I recommend, then you will have to have IPython 2.4 or later installed. IPython is an interactive architecture that provides IPython Notebook, the tool used to write this book. Note that the IPython version has nothing to do with the Python version. IPython 2.4 can run Python 3.4, IPython 3.0 can run Python 2.7, and so on.

I do not cover how to install IPython in this book; requirements change based on what other python installations you may have, whether you use a third party package like Anaconda Python, what operating system you are using, and so on.

The IPython Notebook format was changed as of IPython 3.0. If you are running 2.4 you will still be able to open and run the notebooks, but they will be downconverted for you. If you make changes DO NOT push 2.4 version notebooks to me! I strongly recommend updating to 3.0 as soon as possible, as this format change will just become more frustrating with time.

You will need Python 2.7 or later installed. Almost all of my work is done in Python 3.4, but I periodically test on 2.7. I do not promise any specific check in will work in 2.7 however. I do use Python's "from __future__ import ..." statement to help with compatibility. For example, all prints need to use parenthesis. If you try to add, say, "print 3.14" into the book your script will fail; you must write "print (3.4)" as in Python 3.X.

You will need a recent version of NumPy, SciPy, SymPy, and Matplotlib installed. I don't really know what the minimal version might be. I have NumPy 1.71, SciPy 0.13.0, and Matplotlib 1.4.0 installed on my machines.

Personally, I use the Anaconda Python distribution in all of my work, [available here](https://store.continuum.io/cshop/anaconda/) [3]. I am not selecting them out of favoritism, I am merely documenting my environment. Should you have trouble running any of the code, perhaps knowing this will help you.

Finally, you will need to install FilterPy, described in the next section.

Installation of all of these packages is described in the Installation appendix, which you can read online [here](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix_A_Installation/Appendix_Installation.ipynb).

##Provided Libraries

I am writing an open source Bayesian filtering Python library called **FilterPy**. It is available on github at (https://github.com/rlabbe/filterpy). To ensure that you have the latest release you will want to grab a copy from github, and follow your Python installation's instructions for adding it to the Python search path.

I have also made the project available on PyPi, the Python Package Index. I will be honest, I am not updating this as fast as I am changing the code in the library. That will change as the library and this book mature. To install from PyPi, at the command line issue the command

    pip install filterpy

If you do not have pip, you may follow the instructions here:

https://pip.pypa.io/en/latest/installing.html.


Code that is specific to the book is stored with the book in the subdirectory **code**. This code is in a state of flux; I do not wish to document it here yet. I do mention in the book when I use code from this directory, so it should not be a mystery.

In the *code* subdirectory there are python files with a name like *xxx*_internal.py. I use these to store functions that are useful for a specific chapter. This allows me to hide away Python code that is not particularly interesting to read - I may be generating a plot or chart, and I want you to focus on the contents of the chart, not the mechanics of how I generate that chart with Python. If you are curious as to the mechanics of that, just go and browse the source.

Some chapters introduce functions that are useful for the rest of the book. Those functions are initially defined within the Notebook itself, but the code is also stored in a Python file that is imported if needed in later chapters. I do document when I do this where the function is first defined, but this is still a work in progress. I try to avoid this because then I always face the issue of code in the directory becoming out of sync with the code in the book. However, IPython Notebook does not give us a way to refer to code cells in other notebooks, so this is the only mechanism I know of to share functionality across notebooks.

There is an undocumented directory called **experiments**. This is where I write and test code prior to putting it in the book. There is some interesting stuff in there, and feel free to look at it. As the book evolves I plan to create examples and projects, and a lot of this material will end up there. Small experiments will eventually just be deleted. If you are just interested in reading the book you can safely ignore this directory.


The directory **styles** contains a css file containing the style guide for the book. The default look and feel of IPython Notebook is rather plain. Work is being done on this. I have followed the examples set by books such as [Probabilistic Programming and Bayesian Methods for Hackers](http://nbviewer.ipython.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Chapter1_Introduction.ipynb). I have also been very influenced by Professor Lorena Barba's fantastic work, [available here](https://github.com/barbagroup/CFDPython). I owe all of my look and feel to the work of these projects.

##Thoughts on Python and Coding Math

I am first and foremost a programmer. Most Kalman filtering and other engineering texts are written by mathematicians or engineers. As a result, the software is usually not production quality. I will take Paul Zarchan's book *Fundamentals of Kalman Filtering* as an example. This is a fantastic book and it belongs in your library. But the code is Fortran listing without any subroutines beyond calls to functions like `MATMUL`. This means that Kalman filters are re-implemented over and over again throughout the book. The same listing mixes simulation code with filtering code, so until you become aware of the author's naming style it can be difficult to ascertain what code is the filter and what code is the simulation. Some chapters implement the same filter in subtly different ways, and uses bold text to highlight the few lines that changed. If he needs to use Runge Kutta, that is just embedded in the code, without comments.

There's a better way. If I want to perform an SVD I call `svd`, I do not embed an SVD implementation in my code. This buys me several things. First, I don't have to re-implement SVD multiple times. I don't have to debug SVD several times, and if I do find a bug, I can fix it once and be assured that it now works across all my different projects. Finally, it is readable. It is rare that I care about the implementation of SVD in my projects.

Now, this is a textbook on Kalman filtering, and you could reasonably point out that we *do* care about the implementation of Kalman filters. To an extent that is true, but as you will find out the code that performs the filtering amounts to 7 or so lines of code. The code to implement the math is fairly trivial. Most of the work that Kalman filters requires is the design of the matrices that get fed into the math engine. So that is how I have structured the code.

For example, there is a class named `KalmanFilter` which implements the linear algebra for performing kalman filtering. To use it you will construct an object of that class, initialize various parameters, then enter a while loop where you call `KalmanFilter.predict()` and `KalmanFilter.update()` to incorporate your measurements. Thus most of your programs will be only 20-50 lines, most of that boilerplate - setting up the matrices, and then plotting and/or using the results. The possible downside of that is that the equations that perform the filtering are hidden behind functions, which we could argue is a loss in a pedagogical text. I argue the converse. I want you to learn how to use Kalman filters in the real world, for real projects, and you shouldn't be cutting and pasting established algorithms all over the place. If you want to use ode45 (Runga Kutta) you call that function, you don't re-implement it from scratch. We will do the same here.

However, it is undeniable that you will be a bit less versed with the equations of the Kalman filter as a result. I strongly recommend looking at the source for FilterPy, and maybe even implementing your own filter from scratch to be sure you understand the concepts.

I use a fair number of classes in FilterPy. I do not use inheritance or virtual functions or any of that sort of OO design. I use classes as a way to organize the data that the filters require. For example, the `KalmanFilter` class mentioned above stores matrices called `x`, `P`, `R`, `Q`, and more. I've seen procedural libraries for Kalman filters, and they require the programmer to maintain all of those matrices. This perhaps isn't so bad for a toy program, but start programming, say, a bank of Kalman filters and you will not enjoy having to manage all of those matrices and other associated data.

A word on variable names. I am an advocate for descriptive variable names. `R` is not, normally, descriptive. `R` is the measurement noise covariance matrix, so I could reasonably call it `measurement_noise_covariance`, and I've seen libraries do that. I've chosen not to do that. Why? In the end, Kalman filtering is math. To write a Kalman filter you are going to have to start by sitting down with a piece of paper and doing some math. You will be writing normal algebraic equations. Also, every Kalman filter text and source on the web uses the same linear algebra equations. You cannot read about the Kalman filter without seeing

<span class="math-tex" data-type="tex">\\(\dot{\mathbf{x}} = \mathbf{Fx} + \mathbf{Gu}\\)</span>

in every source (a few sources use A and B instead of F and G). One of my goals in this book is to bring you to the point where you can read the original literature on Kalman filtering. I take an optimistic tone in this book - that Kalman filtering is easy to learn - and in many ways it is. However, for nontrivial problems the difficulty is not the implementation of the equations, but learning how to set up the equations so they solve your problem. In other words, every Kalman filter will implement <span class="math-tex" data-type="tex">\\(\dot{\mathbf{x}} = \mathbf{Fx} + \mathbf{Gu}\\)</span>; the difficult part is figuring out what to put in the matrices <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> and <span class="math-tex" data-type="tex">\\(\mathbf{G}\\)</span> to make your filter work for your problem. Vast amounts of work have been done to figure out how to apply Kalman filters in various domains, and it would be tragic to not be able to read the literature and avail yourself of this research.

So, like it or not you will need to learn that <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> is the *state transition matrix* and that <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span> is the *measurement noise covariance*. Once you know that the code will become readable, and until you know that all publications and web articles on Kalman filters will be inaccessible to you.

Finally I feel that mathematical programming is somewhat different than regular programming; what is readable in one domain is not readable in another. `q = x + m` is opaque in a normal context. On the other hand, `x = .5*a*t**2 + v_0 * t + x_0` is to me the most readable way to write the Newtonian distance equation:

<span class="math-tex" data-type="tex">\\( x = \frac{1}{2}at^2 + v_0 t + x_0\\)</span>

We could write it as

    distance = .5 * constant_acceleration * time_delta**2 +
                initial_velocity * time_delta + initial_distance

but I feel that obscures readability. This is surely debatable for this one equation; but most mathematical programs, and certainly Kalman filters, use systems of equations. I can most easily follow the code, and ensure that it does not have bugs, when it reads as close to the math as possible. Consider this equation taken from the Kalman filter:

<span class="math-tex" data-type="tex">\\( K = PH^T[HPH^T + R]^{-1}\\)</span>

My Python code for this would be

    K = dot3(P, H.T, inv(dot3(H,P,H.T) + R))

It's already a bit hard to read because of the `dot` function calls (required because Python does not yet support an operator for matrix multiplication). But compare this to

    kalman_gain = dot3(apriori_state_covariance, measurement_function_transpose,
                       inverse (dot3(measurement_function, apriori_state_covariance,
                       measurement_function_transpose) +
                       measurement_noise_covariance))

I grant you this version has more context, but I cannot reasonable glance at this and see what math it is implementing. In particular, the linear algebra <span class="math-tex" data-type="tex">\\(HPH^T\\)</span> is doing something very specific - multiplying P by H and its transpose is changing the *basis* of P. It is nearly impossible to see that the Kalman gain is just a ratio of one number divided by a second number which has been converted to a different basis. If you are not solid in linear algebra perhaps that statement does not convey a lot of information to you yet, but I assure you that <span class="math-tex" data-type="tex">\\(K = PH^T[HPH^T + R]^{-1}\\)</span> is saying something very succinctly. There are two key pieces of information here - we are taking a ratio, and we are converting the *basis* of a matrix. I can see that in my first Python line, I cannot see that in the second line.

I will not *win* this argument, and some people will not agree with my naming choices. I will finish by stating, very truthfully, that I made two mistakes the first time I typed that second version and it took me awhile to find it. In any case, I aim for using the mathematical symbol names whenever possible, coupled with readable class and function names. So, it is `KalmanFilter.P`, not `KF.P` and not `KalmanFilter.apriori_state_covariance`.

##License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Kalman Filters and Random Signals in Python</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python" property="cc:attributionName" rel="cc:attributionURL">Roger Labbe</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />

Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python" rel="dct:source">https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python</a>.

## Contact

rlabbejr@gmail.com

##Resources

* [1] http://www.greenteapress.com/
* [2] http://ipython.org/ipython-doc/rel-1.0.0/interactive/nbconvert.html
* [3] https://store.continuum.io/cshop/anaconda/

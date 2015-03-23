[Table of Contents](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

# Designing Kalman Filters

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

## Introduction

In this chapter we will work through the design of several Kalman filters to gain experience and confidence with the various equations and techniques.


For our first multidimensional problem we will track a robot in a 2D space, such as a field. We will start with a simple noisy sensor that outputs noisy <span class="math-tex" data-type="tex">\\((x,y)\\)</span> coordinates which we will need to filter to generate a 2D track. Once we have mastered this concept, we will extend the problem significantly with more sensors and then adding control inputs.
blah blah

## Tracking a Robot

This first attempt at tracking a robot will closely resemble the 1-D dog tracking problem of previous chapters. This will allow us to 'get our feet wet' with Kalman filtering. So, instead of a sensor that outputs position in a hallway, we now have a sensor that supplies a noisy measurement of position in a 2-D space, such as an open field. That is, at each time <span class="math-tex" data-type="tex">\\(T\\)</span> it will provide an <span class="math-tex" data-type="tex">\\((x,y)\\)</span> coordinate pair specifying the measurement of the sensor's position in the field.

Implementation of code to interact with real sensors is beyond the scope of this book, so as before we will program simple simulations in Python to represent the sensors. We will develop several of these sensors as we go, each with more complications, so as I program them I will just append a number to the function name. `pos_sensor1()` is the first sensor we write, and so on.

So let's start with a very simple sensor, one that travels in a straight line. It takes as input the last position, velocity, and how much noise we want, and returns the new position.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy.random as random
import copy
class PosSensor1(object):
    def __init__(self, pos = [0,0], vel = (0,0), noise_scale = 1.):
        self.vel = vel
        self.noise_scale = noise_scale
        self.pos = copy.deepcopy(pos)

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [self.pos[0] + random.randn() * self.noise_scale,
                self.pos[1] + random.randn() * self.noise_scale]
</pre>

A quick test to verify that it works as we expect.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import book_plots as bp

pos = [4,3]
sensor = PosSensor1 (pos, (2,1), 1)

for i in range (50):
    pos = sensor.read()
    bp.plot_measurements(pos[0], pos[1])

plt.show()
</pre>

That looks correct. The slope is 1/2, as we would expect with a velocity of (2,1), and the data seems to start at near (6,4).

### Step 1: Choose the State Variables

As always, the first step is to choose our state variables. We are tracking in two dimensions and have a sensor that gives us a reading in each of those two dimensions, so we  know that we have the two *observed variables* <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span>. If we created our Kalman filter using only those two variables the performance would not be very good because we would be ignoring the information velocity can provide to us. We will want to incorporate velocity into our equations as well. I will represent this as

<span class="math-tex" data-type="tex">\\(\mathbf{x} =
\begin{bmatrix}x\\v_x\\y\\v_y\end{bmatrix}\\)</span>

There is nothing special about this organization. I could have listed the (xy) coordinates first followed by the velocities, and/or I could done this as a row matrix instead of a column matrix. For example, I could have chosen:

<span class="math-tex" data-type="tex">\\(\mathbf{x} =
\begin{bmatrix}x&y&v_x&v_y\end{bmatrix}\\)</span>

All that matters is that the rest of my derivation uses this same scheme. However, it is typical to use column matrices for state variables, and I prefer it, so that is what we will use.

It might be a good time to pause and address how you identify the unobserved variables. This particular example is somewhat obvious because we already worked through the 1D case in the previous chapters. Would it be so obvious if we were filtering market data, population data from a biology experiment, and so on? Probably not. There is no easy answer to this question. The first thing to ask yourself is what is the interpretation of the first and second derivatives of the data from the sensors. We do that because obtaining the first and second derivatives is mathematically trivial if you are reading from the sensors using a fixed time step. The first derivative is just the difference between two successive readings. In our tracking case the first derivative has an obvious physical interpretation: the difference between two successive positions is velocity.

Beyond this you can start looking at how you might combine the data from two or more different sensors to produce more information. This opens up the field of *sensor fusion*, and we will be covering examples of this in later sections. For now, recognize that choosing the appropriate state variables is paramount to getting the best possible performance from your filter.

### **Step 2:** Design State Transition Function

Our next step is to design the state transition function. Recall that the state transition function is implemented as a matrix <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> that we multiply with the previous state of our system to get the next state, like so.

<span class="math-tex" data-type="tex">\\(\mathbf{x}' = \mathbf{Fx}\\)</span>

I will not belabor this as it is very similar to the 1-D case we did in the previous chapter. Our state equations for position and velocity would be:

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
x' &= (1*x) + (\Delta t * v_x) + (0*y) + (0 * v_y) \\
v_x &= (0*x) +  (1*v_x) + (0*y) + (0 * v_y) \\
y' &= (0*x) + (0* v_x)         + (1*y) + (\Delta t * v_y) \\
v_y &= (0*x) +  (0*v_x) + (0*y) + (1 * v_y)
\end{aligned}
\\)</span>

Laying it out that way shows us both the values and row-column organization required for <span class="math-tex" data-type="tex">\\(\small\mathbf{F}\\)</span>. In linear algebra, we would write this as:

<span class="math-tex" data-type="tex">\\(
\begin{bmatrix}x\\v_x\\y\\v_y\end{bmatrix}' = \begin{bmatrix}1& \Delta t& 0& 0\\0& 1& 0& 0\\0& 0& 1& \Delta t\\ 0& 0& 0& 1\end{bmatrix}\begin{bmatrix}x\\v_x\\y\\v_y\end{bmatrix}\\)</span>

So, let's do this in Python. It is very simple; the only thing new here is setting `dim_z` to 2. We will see why it is set to 2 in step 4.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from filterpy.kalman import KalmanFilter
import numpy as np

tracker = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.   # time step

tracker.F = np.array ([[1, dt, 0,  0],
                      [0,  1, 0,  0],
                      [0,  0, 1, dt],
                      [0,  0, 0,  1]])
</pre>

### **Step 3**: Design the Motion Function

We have no control inputs to our robot (yet!), so this step is trivial - set the motion input <span class="math-tex" data-type="tex">\\(\small\mathbf{u}\\)</span> to zero. This is done for us by the class when it is created so we can skip this step, but for completeness we will be explicit.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tracker.u = 0.
</pre>

### **Step 4**: Design the Measurement Function

The measurement function defines how we go from the state variables to the measurements using the equation <span class="math-tex" data-type="tex">\\(\mathbf{z} = \mathbf{Hx}\\)</span>. At first this is a bit counterintuitive, after all, we use the Kalman filter to go from measurements to state. But the update step needs to compute the residual between the current measurement and the measurement represented by the prediction step. Therefore <span class="math-tex" data-type="tex">\\(\textbf{H}\\)</span> is multiplied by the state <span class="math-tex" data-type="tex">\\(\textbf{x}\\)</span> to produce a measurement <span class="math-tex" data-type="tex">\\(\textbf{z}\\)</span>.

In this case we have measurements for (x,y), so <span class="math-tex" data-type="tex">\\(\textbf{z}\\)</span> must be of dimension <span class="math-tex" data-type="tex">\\(2\times 1\\)</span>. Our state variable is size <span class="math-tex" data-type="tex">\\(4\times 1\\)</span>. We can deduce the required size for <span class="math-tex" data-type="tex">\\(\textbf{H}\\)</span> by recalling that multiplying a matrix of size <span class="math-tex" data-type="tex">\\(m\times n\\)</span> by <span class="math-tex" data-type="tex">\\(n\times p\\)</span> yields a matrix of size <span class="math-tex" data-type="tex">\\(m\times p\\)</span>. Thus,

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
(2\times 1) &= (a\times b)(4 \times 1) \\
&= (a\times 4)(4\times 1) \\
&= (2\times 4)(4\times 1)
\end{aligned}\\)</span>

So, <span class="math-tex" data-type="tex">\\(\textbf{H}\\)</span> is of size <span class="math-tex" data-type="tex">\\(2\times 4\\)</span>.

Filling in the values for <span class="math-tex" data-type="tex">\\(\textbf{H}\\)</span> is easy in this case because the measurement is the position of the robot, which is the <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> variables of the state <span class="math-tex" data-type="tex">\\(\textbf{x}\\)</span>. Let's make this just slightly more interesting by deciding we want to change units. So we will assume that the measurements are returned in feet, and that we desire to work in meters. Converting from feet to meters is a simple as multiplying by 0.3048. However, we are converting from state (meters) to measurements (feet) so we need to divide by 0.3048. So

<span class="math-tex" data-type="tex">\\(\mathbf{H} =
\begin{bmatrix}
\frac{1}{0.3048} & 0 & 0 & 0 \\
0 & 0 & \frac{1}{0.3048} & 0
\end{bmatrix}
\\)</span>

which corresponds to these linear equations
<span class="math-tex" data-type="tex">\\(
\begin{aligned}
z_x' &= (\frac{x}{0.3048}) + (0* v_x) + (0*y) + (0 * v_y) \\
z_y' &= (0*x) + (0* v_x) + (\frac{y}{0.3048}) + (0 * v_y) \\
\end{aligned}
\\)</span>

To be clear about my intentions here, this is a pretty simple problem, and we could have easily found the equations directly without going through the dimensional analysis that I did above. In fact, an earlier draft did just that. But it is useful to remember that the equations of the Kalman filter imply a specific dimensionality for all of the matrices, and when I start to get lost as to how to design something it is often extremely useful to look at the matrix dimensions. Not sure how to design <span class="math-tex" data-type="tex">\\(\textbf{H}\\)</span>?
Here is the Python that implements this:

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tracker.H = np.array([[1/0.3048, 0, 0, 0],
                      [0, 0, 1/0.3048, 0]])
print(tracker.H)
</pre>

### **Step 5**: Design the Measurement Noise Matrix

In this step we need to mathematically model the noise in our sensor. For now we will make the simple assumption that the <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> variables are independent Gaussian processes. That is, the noise in x is not in any way dependent on the noise in y, and the noise is normally distributed about the mean. For now let's set the variance for <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> to be 5 for each. They are independent, so there is no covariance, and our off diagonals will be 0. This gives us:

<span class="math-tex" data-type="tex">\\(\mathbf{R} = \begin{bmatrix}5&0\\0&5\end{bmatrix}\\)</span>

It is a <span class="math-tex" data-type="tex">\\(2{\times}2\\)</span> matrix because we have 2 sensor inputs, and covariance matrices are always of size <span class="math-tex" data-type="tex">\\(n{\times}n\\)</span> for <span class="math-tex" data-type="tex">\\(n\\)</span> variables. In Python we write:

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tracker.R = np.array([[5, 0],
                      [0, 5]])
print(tracker.R)
</pre>

### Step 6: Design the Process Noise Matrix

Finally, we design the process noise. We don't yet have a good way to model process noise, so for now we will assume there is a small amount of process noise, say 0.1 for each state variable. Later we will tackle this admittedly difficult topic in more detail. We have 4 state variables, so we need a <span class="math-tex" data-type="tex">\\(4{\times}4\\)</span> covariance matrix:

<span class="math-tex" data-type="tex">\\(\mathbf{Q} = \begin{bmatrix}0.1&0&0&0\\0&0.1&0&0\\0&0&0.1&0\\0&0&0&0.1\end{bmatrix}\\)</span>

In Python I will use the numpy eye helper function to create an identity matrix for us, and multiply it by 0.1 to get the desired result.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tracker.Q = np.eye(4) * 0.1
print(tracker.Q)
</pre>

### **Step 7**: Design Initial Conditions

For our simple problem we will set the initial position at (0,0) with a velocity of (0,0). Since that is a pure guess, we will set the covariance matrix <span class="math-tex" data-type="tex">\\(\small\mathbf{P}\\)</span> to a large value.
<span class="math-tex" data-type="tex">\\( \mathbf{x} = \begin{bmatrix}0\\0\\0\\0\end{bmatrix}\\
\mathbf{P} = \begin{bmatrix}500&0&0&0\\0&500&0&0\\0&0&500&0\\0&0&0&500\end{bmatrix}\\)</span>

In Python we implement that with

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tracker.x = np.array([[0, 0, 0, 0]]).T
tracker.P = np.eye(4) * 500.
print(tracker.x)
print()
print (tracker.P)
</pre>

## Implement the Filter Code

Design is complete, now we just have to write the Python code to run our filter, and output the data in the format of our choice. To keep the code clear, let's just print a plot of the track. We will run the code for 30 iterations.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tracker = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0   # time step

tracker.F = np.array ([[1, dt, 0,  0],
                       [0,  1, 0,  0],
                       [0,  0, 1, dt],
                       [0,  0, 0,  1]])
tracker.u = 0.
tracker.H = np.array ([[1/0.3048, 0, 0, 0],
                       [0, 0, 1/0.3048, 0]])

tracker.R = np.eye(2) * 5
tracker.Q = np.eye(4) * .1

tracker.x = np.array([[0,0,0,0]]).T
tracker.P = np.eye(4) * 500.

# initialize storage and other variables for the run
count = 30
xs, ys = [],[]
pxs, pys = [],[]

sensor = PosSensor1 ([0,0], (2,1), 1.)

for i in range(count):
    pos = sensor.read()
    z = np.array([[pos[0]], [pos[1]]])

    tracker.predict ()
    tracker.update (z)

    xs.append (tracker.x[0,0])
    ys.append (tracker.x[2,0])
    pxs.append (pos[0]*.3048)
    pys.append(pos[1]*.3048)

bp.plot_filter(xs, ys)
bp.plot_measurements(pxs, pys)
plt.legend(loc=2)
plt.xlim((0,20))
plt.show()
</pre>

I encourage you to play with this, setting <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> and <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span> to various values.  However, we did a fair amount of that sort of thing in the last chapters, and we have a lot of material to cover, so I will move on to more complicated cases where we will also have a chance to experience changing these values.

Now I will run the same Kalman filter with the same settings, but also plot the covariance ellipse for <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span>. First, the code without explanation, so we can see the output. I print the last covariance to see what it looks like. But before you scroll down to look at the results, what do you think it will look like? You have enough information to figure this out but this is still new to you, so don't be discouraged if you get it wrong.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import stats

tracker = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0   # time step

tracker.F = np.array([[1, dt, 0,  0],
                      [0,  1, 0,  0],
                      [0,  0, 1, dt],
                      [0,  0, 0,  1]])
tracker.u = 0.
tracker.H = np.array([[1/0.3048, 0, 0, 0],
                      [0, 0, 1/0.3048, 0]])

tracker.R = np.eye(2) * 5
tracker.Q = np.eye(4) * .1

tracker.x = np.array([[0, 0, 0, 0]]).T
tracker.P = np.eye(4) * 500.

# initialize storage and other variables for the run
count = 30
xs, ys = [], []
pxs, pys = [], []

sensor = PosSensor1([0,0], (2,1), 1.)

for i in range(count):
    pos = sensor.read()
    z = np.array([[pos[0]], [pos[1]]])

    tracker.predict()
    tracker.update(z)

    xs.append(tracker.x[0,0])
    ys.append(tracker.x[2,0])
    pxs.append(pos[0]*.3048)
    pys.append(pos[1]*.3048)

    # plot covariance of x and y
    cov = np.array([[tracker.P[0,0], tracker.P[2,0]],
                    [tracker.P[0,2], tracker.P[2,2]]])

    stats.plot_covariance_ellipse(
        (tracker.x[0,0], tracker.x[2,0]), cov=cov,
         facecolor='g', alpha=0.15)


bp.plot_filter(xs, ys)
bp.plot_measurements(pxs, pys)
plt.legend(loc=2)
plt.show()
print("final P is:")
print(tracker.P)
</pre>

Did you correctly predict what the covariance matrix and plots would look like? Perhaps you were expecting a tilted ellipse, as in the last chapters. If so, recall that in those chapters we were not plotting <span class="math-tex" data-type="tex">\\(x\\)</span> against <span class="math-tex" data-type="tex">\\(y\\)</span>, but <span class="math-tex" data-type="tex">\\(x\\)</span> against <span class="math-tex" data-type="tex">\\(\dot{x}\\)</span>. <span class="math-tex" data-type="tex">\\(x\\)</span> *is correlated* to <span class="math-tex" data-type="tex">\\(\dot{x}\\)</span>, but <span class="math-tex" data-type="tex">\\(x\\)</span> is not correlated or dependent on <span class="math-tex" data-type="tex">\\(y\\)</span>. Therefore our ellipses are not tilted. Furthermore, the noise for both <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> are modeled to have the same value, 5, in <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span>. If we were to set R to, for example,

<span class="math-tex" data-type="tex">\\(\mathbf{R} = \begin{bmatrix}10&0\\0&5\end{bmatrix}\\)</span>

we would be telling the Kalman filter that there is more noise in <span class="math-tex" data-type="tex">\\(x\\)</span> than <span class="math-tex" data-type="tex">\\(y\\)</span>, and our ellipses would be longer than they are tall.

The final P tells us everything we need to know about the correlation between the state variables. If we look at the diagonal alone we see the variance for each variable. In other words <span class="math-tex" data-type="tex">\\(\mathbf{P}_{0,0}\\)</span> is the variance for x, <span class="math-tex" data-type="tex">\\(\mathbf{P}_{1,1}\\)</span> is the variance for <span class="math-tex" data-type="tex">\\(\dot{x}\\)</span>, <span class="math-tex" data-type="tex">\\(\mathbf{P}_{2,2}\\)</span> is the variance for y, and <span class="math-tex" data-type="tex">\\(\mathbf{P}_{3,3}\\)</span> is the variance for <span class="math-tex" data-type="tex">\\(\dot{y}\\)</span>. We can extract the diagonal of a matrix using *numpy.diag()*.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
print(np.diag(tracker.P))
</pre>

The covariance matrix contains four <span class="math-tex" data-type="tex">\\(2{\times}2\\)</span> matrices that you should be able to easily pick out. This is due to the correlation of <span class="math-tex" data-type="tex">\\(x\\)</span> to <span class="math-tex" data-type="tex">\\(\dot{x}\\)</span>, and of <span class="math-tex" data-type="tex">\\(y\\)</span> to <span class="math-tex" data-type="tex">\\(\dot{y}\\)</span>. The upper left hand side shows the covariance of <span class="math-tex" data-type="tex">\\(x\\)</span> to <span class="math-tex" data-type="tex">\\(\dot{x}\\)</span>. Let's extract and print, and plot it.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
c = tracker.P[0:2, 0:2]
print(c)
stats.plot_covariance_ellipse((0, 0), cov=c, facecolor='g', alpha=0.2)
</pre>

The covariance contains the data for <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(\dot{x}\\)</span> in the upper left because of how it is organized. Recall that entries <span class="math-tex" data-type="tex">\\(\mathbf{P}_{i,j}\\)</span> and <span class="math-tex" data-type="tex">\\(\mathbf{P}_{j,i}\\)</span> contain <span class="math-tex" data-type="tex">\\(p\sigma_1\sigma_2\\)</span>.

Finally, let's look at the lower left side of <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span>, which is all 0s. Why 0s? Consider <span class="math-tex" data-type="tex">\\(\mathbf{P}_{3,0}\\)</span>. That stores the term <span class="math-tex" data-type="tex">\\(p\sigma_3\sigma_0\\)</span>, which is the covariance between <span class="math-tex" data-type="tex">\\(\dot{y}\\)</span> and <span class="math-tex" data-type="tex">\\(x\\)</span>. These are independent, so the term will be 0. The rest of the terms are for similarly independent variables.

## The Effect of Order

So far in this book we have only really considered tracking position and velocity. That has worked well, but only because I have been carefully selecting problems for which this was an appropriate choice. You know have enough experience with the Kalman filter to consider this in more general terms.

What do I mean by "order"? In the context of these system models it is the number of derivatives required to accurately model a system. Consider a system that does not change, such as the height of a building. There is no change, so there is no need for a derivative, and the order of the system is zero. We could express this in an equation as

<span class="math-tex" data-type="tex">\\(x = 312.5\\)</span>

A first order system has a first derivative. For example, change of position is velocity, and we can write this as

<span class="math-tex" data-type="tex">\\( v = \frac{dx}{dt}\\)</span>

which we integrate into the Newton equation
<span class="math-tex" data-type="tex">\\( x = vt + x_0.\\)</span>

This is also called a *constant velocity* model, because of the assumption of a constant velocity.

So a second order has a second derivative. The second derivative of position is acceleration, with the equation

<span class="math-tex" data-type="tex">\\(a = \frac{d^2x}{dt^2}\\)</span>

which we integrate into

<span class="math-tex" data-type="tex">\\( x = \frac{1}{2}at^2 +v_0t + x_0.\\)</span>

This is also known as a *constant acceleration* model.

Another, equivalent way of looking at this is to consider the order of the polynomial. The constant acceleration model has a second derivative, so it is second order. Likewise, the polynomial <span class="math-tex" data-type="tex">\\(x = \frac{1}{2}at^2 +v_0t + x_0\\)</span> is second order.

When we design the state variables and process model we must choose the order of the system we want to model. Let's say we are tracking something with a constant velocity. No real world process is perfect, and so there will be slight variations in the velocity over short time period. You might reason that the best approach is to use a second order filter, allowing the acceleration term to deal with the slight variations in velocity.

That doesn't work as nicely as you might think. To thoroughly understand this issue lets see the effects of using a  process model that does not match the order of the system being filtered.

First we need a system to filter. I'll write a class to simulate on object with a constant velocity. Essentially no physical system has a truly constant velocity, so on each update we alter the velocity by a small amount. I also write a sensor to simulate Gaussian noise in a sensor. The code is below, and a plot an example run to verify that it is working correctly.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from book_plots import plot_track

np.random.seed(124)
class ConstantVelocityObject(object):
    def __init__(self, x0=0, vel=1., noise_scale=0.06):
        self.x = x0
        self.vel = vel
        self.noise_scale = noise_scale


    def update(self):
        self.vel += randn()*self.noise_scale
        self.x += self.vel
        return (self.x, self.vel)


def sense(x, noise_scale=1.):
    return x[0] + randn()*noise_scale


obj = ConstantVelocityObject()

xs = []
zs = []
for i in range(50):
    x = obj.update()
    z = sense(x)
    xs.append(x)
    zs.append(z)


xs = np.asarray(xs)
bp.plot_track(xs[:,0])
bp.plot_measurements(range(50), zs)
plt.legend(loc='best')
plt.show()
</pre>

I am satisfied with this plot. The track is not perfectly straight due to the noise that we added to the system - this could be the track of a person walking down the street, or perhaps of an aircraft being buffeted by variable winds. There is no *intentional* acceleration here, so we call it a constant velocity system. Again, you may be asking yourself that since there is in fact a tiny bit of acceleration going on why would we not use a second order Kalman filter to account for those changes? Let's find out.

How does one design a zero order, first order, or second order Kalman filter. We have been doing it all along, but just not using those terms. It might be slightly tedious, but I will elaborate fully on each - if the concept is clear to you feel free to skim a bit. However, I think that reading carefully will really cement the idea of filter order in your mind.

### Zero Order Kalman Filter

A zero order Kalman filter is just a filter that tracks with no derivatives. We are tracking position, so that means we only have a state variable for position (no velocity or acceleration), and the state transition function also only accounts for position. Using the matrix formulation we would say that the state variable is

<span class="math-tex" data-type="tex">\\(\mathbf{x} = \begin{bmatrix}x\end{bmatrix}\\)</span>

The state transition function is very simple. There is no change in position, so we need to model <span class="math-tex" data-type="tex">\\(x=x\\)</span>; in other words, *x* at time t+1 is the same as it was at time t. In matrix form, our state transition function is

<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}1\end{bmatrix}\\)</span>

The measurement function is very easy. Recall that we need to define how to convert the state variable <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> *into* a measurement. We will assume that our measurements are positions. The state variable only contains a position, so we get

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}1\end{bmatrix}\\)</span>


That is pretty much it. Let's write a function that constructs and returns a zero order Kalman filter to us.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def ZeroOrderKF(R, Q):
    """ Create zero order Kalman filter. Specify R and Q as floats."""
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([0.])
    kf.R *= R
    kf.Q *= Q
    kf.P *= 20
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    return kf
</pre>

### First Order Kalman Filter

A first order Kalman filter tracks a first order system, such as position and velocity. We already did this for the dog tracking problem above, so this should be very clear. But let's do it again.

A first order system has position and velocity, so the state variable needs both of these. The matrix formulation could be

<span class="math-tex" data-type="tex">\\( \mathbf{x} = \begin{bmatrix}x\\\dot{x}\end{bmatrix}\\)</span>

As an aside, there is nothing stopping us from choosing

<span class="math-tex" data-type="tex">\\( \mathbf{x} = \begin{bmatrix}\dot{x}\\x\end{bmatrix}\\)</span>

but all texts and software that I know of choose the first form as more natural. You would just have to design the rest of the matrices to take this ordering into account.

So now we have to design our state transition. The Newtonian equations for a time step are:

<span class="math-tex" data-type="tex">\\(\begin{aligned} x_t &= x_{t-1} + v\Delta t \\
 v_t &= v_{t-1}\end{aligned}\\)</span>

Recall that we need to convert this into the linear equation

<span class="math-tex" data-type="tex">\\(\begin{bmatrix}x\\\dot{x}\end{bmatrix} = \mathbf{F}\begin{bmatrix}x\\\dot{x}\end{bmatrix}\\)</span>

Setting

<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}1 &\Delta t\\ 0 & 1\end{bmatrix}\\)</span>

gives us the equations above. If this is not clear, work out the matrix multiplication:

<span class="math-tex" data-type="tex">\\( x = 1x + dt \dot{x} \\
\dot{x} = 0x + 1\dot{x}\\)</span>

Finally, we design the measurement function. The measurement function needs to implement

<span class="math-tex" data-type="tex">\\(z = \mathbf{Hx}\\)</span>

Our sensor still only reads position, so it should take the position from the state, and 0 out the velocity, like so:

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}1 & 0 \end{bmatrix}\\)</span>

As in the previous section we will define a function that constructs and returns a Kalman filter that implements these equations.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from filterpy.common import Q_discrete_white_noise

def FirstOrderKF(R, Q, dt):
    """ Create zero order Kalman filter. Specify R and Q as floats."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.zeros(2)
    kf.P *= np.array([[100,0], [0,1]])
    kf.R *= R
    kf.Q = Q_discrete_white_noise(2, dt, Q)
    kf.F = np.array([[1., dt],
                     [0. , 1]])
    kf.H = np.array([[1., 0]])
    return kf
</pre>

### Second Order Kalman Filter

A second order Kalman filter tracks a second order system, such as position, velocity and acceleration. The state variables will need to contain all three. The matrix formulation could be

<span class="math-tex" data-type="tex">\\( \mathbf{x} = \begin{bmatrix}x\\\dot{x}\\\ddot{x}\end{bmatrix}\\)</span>


So now we have to design our state transition. The Newtonian equations for a time step are:

<span class="math-tex" data-type="tex">\\(\begin{aligned} x_t &= x_{t-1} + v_{t-1}\Delta t + 0.5a_{t-1} \Delta t^2 \\
 v_t &= v_{t-1} \Delta t + a_{t-1} \\
 a_t &= a_{t-1}\end{aligned}\\)</span>

Recall that we need to convert this into the linear equation

<span class="math-tex" data-type="tex">\\(\begin{bmatrix}x\\\dot{x}\\\ddot{x}\end{bmatrix} = \mathbf{F}\begin{bmatrix}x\\\dot{x}\\\ddot{x}\end{bmatrix}\\)</span>

Setting

<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}1 & \Delta t &.5\Delta t^2\\
0 & 1 & \Delta t \\
0 & 0 & 1\end{bmatrix}\\)</span>

gives us the equations above.

Finally, we design the measurement function. The measurement function needs to implement

<span class="math-tex" data-type="tex">\\(z = \mathbf{Hx}\\)</span>

Our sensor still only reads position, so it should take the position from the state, and 0 out the velocity, like so:

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}1 & 0 & 0\end{bmatrix}\\)</span>

As in the previous section we will define a function that constructs and returns a Kalman filter that implements these equations.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def SecondOrderKF(R_std, Q, dt):
    """ Create zero order Kalman filter. Specify R and Q as floats."""
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.zeros(3)
    kf.P[0,0] = 100
    kf.P[1,1] = 1
    kf.P[2,2] = 1
    kf.R *= R_std**2
    kf.Q = Q_discrete_white_noise(3, dt, Q)
    kf.F = np.array([[1., dt, .5*dt*dt],
                     [0., 1.,       dt],
                     [0., 0.,       1.]])
    kf.H = np.array([[1., 0., 0.]])
    return kf
</pre>

### Evaluating the Performance

We have implemented the Kalman filters and the simulated first order system, so now we can run each Kalman filter against the simulation and evaluate the results.

How do we evaluate the results? We can do this qualitatively by plotting the track and the Kalman filter output and eyeballing the results. However, we can do this far more rigorously with mathematics. Recall that system covariance matrix <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> contains the computed variance and covariances for each of the state variables. The diagonal contains the variance. If you think back to the Gaussian chapter you'll remember that roughly 99% of all measurements fall within three standard deviations if the noise is Gaussian, and, of course, the standard deviation can be computed as the square root of the variance. If this is not clear please review the Gaussian chapter before continuing, as this is an important point.

So we can evaluate the filter by looking at the residuals between the estimated state and actual state and comparing them to the standard deviations which we derive from <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span>. If the filter is performing correctly 99% of the residuals will fall within the third standard deviation. This is true for all the state variables, not just for the position.

So let's run the first order Kalman filter against our first order system and access it's performance. You can probably guess that it will do well, but let's look at it using the standard deviations.

First, let's write a routine to generate the noisy measurements for us.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def simulate_system(Q, count):
    obj = ConstantVelocityObject(x0=0, vel=1, noise_scale=Q)
    zs = []
    xs = []
    for i in range(count):
        x = obj.update()
        z = sense(x)
        xs.append(x)
        zs.append(z)
    return np.asarray(xs), zs
</pre>

And now a routine to perform the filtering.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def filter_data(kf, zs):
    # save output for plotting
    fxs = []
    ps = []

    for z in zs:
        kf.predict()
        kf.update(z)

        fxs.append(kf.x)
        ps.append(kf.P.diagonal())

    fxs = np.asarray(fxs)
    ps = np.asarray(ps)

    return fxs, ps
</pre>

And to plot the track results.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def plot_kf_output(xs, filter_xs, zs, title=None):
    bp.plot_filter(filter_xs[:,0])
    bp.plot_track(xs[:,0])

    if zs is not None:
        bp.plot_measurements(zs)
    plt.legend(loc='best')
    plt.ylabel('meters')
    plt.xlabel('time (sec)')
    if title is not None:
        plt.title(title)
    plt.xlim((-1, len(xs)))
    plt.ylim((-1, len(xs)))
    #plt.axis('equal')
    plt.show()
</pre>

Now we are prepared to run the filter and look at the results.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
R = 1
Q = 0.03

xs, zs = simulate_system(Q=Q, count=50)

kf = FirstOrderKF(R, Q, dt=1)
fxs1, ps1 = filter_data(kf, zs)

plt.figure()
plot_kf_output(xs, fxs1, zs)
</pre>

It looks like the filter is performing well, but it is hard to tell exactly how well. Let's look at the residuals and see if they help. You may have noticed that in the code above I saved the covariance at each step. I did that to use in the following plot. The ConstantVelocityObject class returns a tuple of (position, velocity) for the real object, and this is stored in the array `xs`, and the filter's estimates are in `fxs`.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def plot_residuals(xs, filter_xs, Ps, title, y_label):
    res = xs - filter_xs
    plt.plot(res)
    bp.plot_residual_limits(Ps)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('time (sec)')

    plt.show()
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plot_residuals(xs[:,0], fxs1[:,0], ps1[:,0],
               'First Order Position Residuals',
               'meters')
</pre>

How do we interpret this plot? The residual is drawn as the jagged line - the difference between the measurement and the actual position. If there was no measurement noise and the Kalman filter prediction was always perfect the residual would always be zero. So the ideal output would be a horizontal line at 0. We can see that the residual is centered around 0, so this gives us confidence that the noise is Gaussian (because the errors fall equally above and below 0). The yellow area between dotted lines show the theoretical performance of the filter for 1 standard deviations. In other words, approximately 68% of the errors should fall within the dotted lines. The residual falls within this range, so we see that the filter is performing well, and that it is not diverging.

But that is just for position. Let's look at the residuals for velocity.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plot_residuals(xs[:,1], fxs1[:,1], ps1[:,1],
               'First Order Velocity Residuals',
               'meters/sec')
</pre>

Again, as expected, the residual falls within the theoretical performance of the filter, so we feel confident that the filter is well designed for this system.


Now let's do the same thing using the zero order Kalman filter. All of the code and math is largely the same, so let's just look at the results without discussing the implementation much.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
kf0 = ZeroOrderKF(R, Q)
fxs0, ps0 = filter_data(kf0, zs)

plot_kf_output(xs, fxs0, zs)
</pre>

As we would expect, the filter has problems. Think back to the g-h filter, where we incorporated acceleration into the system. The g-h filter always lagged the input because there were not enough terms to allow the filter to adjust quickly enough to the changes in velocity. The same thing is happening here, just one order lower. On every `predict()` step the Kalman filter assumes that there is no change in position - if the current position is 4.3 it will predict that the position at the next time period is 4.3. Of course, the actual position is closer to 5.3. The measurement, with noise, might be 5.4, so the filter chooses an estimate part way between 4.3 and 5.4, causing it to lag the actual value of 5.3 by a significant amount. This same thing happens in the next step, the next one, and so on. The filter never catches up.

Now let's look at the residuals. We are not tracking velocity, so we can only look at the residual for position.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plot_residuals(xs[:,0], fxs0[:,0], ps0[:,0],
               'Zero Order Position Residuals',
               'meters')
</pre>

We can see that the filter diverges almost immediately. After the first second the residual exceeds the bounds of three standard deviations. It is important to understand that the covariance matrix <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> is only reporting the *theoretical* performance of the filter *assuming* all of the inputs are correct. In other words, this Kalman filter is diverging, but <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> implies that the Kalman filter's estimates are getting better and better with time because the variance is getting smaller. The filter has no way to know that you are lying to it about the system.

In this system the divergence is immediate and striking. In many systems it will only be gradual, and/or slight. It is important to look at charts like these for your systems to ensure that the performance of the filter is within the bounds of its theoretical performance.

Now let's try a third order system. This might strike you as a good thing to do. After all, we know there is a bit of noise in the movement of the simulated object, which implies there is some acceleration. Why not model the acceleration with a second order model. If there is no acceleration, the acceleration should just be estimated to be 0. But is that what happens? Think about it before going on.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
kf2 = SecondOrderKF(R, Q, dt=1)
fxs2, ps2 = filter_data(kf2, zs)

plot_kf_output(xs, fxs2, zs)
</pre>

Did this perform as you expected? We can see that even though the system does have a slight amount of acceleration in it the second order filter performs poorly compared to the first order filter. Why does this? The system believes that there is acceleration in the system, and so the large changes in the measurement gets interpreted as acceleration instead of noise. Thus you can see that the filter tracks the noise in the system quite closely. Not only that, but it *overshoots* the noise in places if the noise is consistently above or below the track because the filter incorrectly assumes an acceleration that does not exist, and so it's prediction goes further and further away from the track on each measurement. This is not a good state of affairs.

Still, the track doesn't look *horrible*. Let's see the story that the residuals tell. I will add a wrinkle here. The residuals for the order 2 system do not look terrible in that they do not diverge or exceed three standard deviations. However, it is very telling to look at the residuals for the first order vs the second order filter, so I have plotted both on the same graph.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
res = xs[:,0] - fxs2[:,0]
res1 = xs[:,0] - fxs1[:,0]

plt.plot(res1, ls="--", label='order 1')
plt.plot(res, label='order 2')
bp.plot_residual_limits(ps2[:,0])
plt.title('Second Order Position Residuals')
plt.legend()
plt.ylabel('meters')
plt.xlabel('time (sec)')
plt.show()
</pre>

We can see that the residuals for the second order filter fall nicely within the theoretical limits of the filter. When we compare them against the first order residuals we may conclude that the second order is slight worse, but the difference is not large. There is nothing very alarming here.

Now let's look at the residuals for the velocity.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
res = xs[:,1] - fxs2[:,1]
res1 = xs[:,1] - fxs1[:,1]
plt.plot(res, label='order 2')
plt.plot(res1, ls='--', label='order 1')
bp.plot_residual_limits(ps2[:,1])
plt.title('Second Order Velocity Residuals')
plt.legend()
plt.ylabel('meters/sec')
plt.xlabel('time (sec)')
plt.show()
</pre>

Here the story is very different. While the residuals of the second order system fall within the theoretical bounds of the filter's performance, we can see that the residuals are *far* worse than for the first order filter. This is the usual result this scenario. The filter is assuming that there is acceleration that does not exist. It mistakes noise in the measurement as acceleration and this gets added into the velocity estimate on every predict cycle. Of course the acceleration is not actually there and so the residual for the velocity is much larger than it optimum.

I have one more trick up my sleeve. We have a first order system; i.e. the velocity is more-or-less constant. Real world systems are never perfect, so of course the velocity is never exactly the same between time periods. When we use a first order filter we account for that slight variation in velocity with the *process noise*. The matrix <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> is computed to account for this slight variation. If we move to a second order filter we are now accounting for the changes in velocity. Perhaps now we have no process noise, and we can set <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> to zero!

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
kf2 = SecondOrderKF(R, 0, dt=1)
fxs2, ps2 = filter_data(kf2, zs)

plot_kf_output(xs, fxs2, zs)
</pre>

To my eye that looks quite good! The filter quickly converges to the actual track. Success!

Or, maybe not. Setting the process noise to 0 tells the filter that the process model is perfect. I've yet to hear of a perfect physical system. Let's look at the performance of the filter over a longer period of time.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
np.random.seed(25944)
xs500, zs500 = simulate_system(Q=Q, count=500)

kf2 = SecondOrderKF(R, 0, dt=1)
fxs2, ps2 = filter_data(kf2, zs500)

plot_kf_output(xs500, fxs2, zs500)
plot_residuals(xs500[:,0], fxs2[:,0], ps2[:,0],
               'Zero Order Position Residuals',
               'meters')
</pre>

We can see that the performance of the filter is abysmal. We can see that in the track plot where the filter diverges from the track for an extended period of time. The divergence may or may not seem large to you. The residual plot makes the problem more apparent. Just before the 100th update the filter diverges sharply from the theoretical performance. It *might* be converging at the end, but I doubt it.

Why is this happening? Recall that if we set the process noise to zero we are telling the filter to use only the process model. The measurements end up getting ignored. The physical system is *not* perfect, and so the filter is unable to adapt to this imperfect behavior.

Maybe just a really low process noise? Let's try that.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
np.random.seed(32594)
xs2000, zs2000 = simulate_system(Q=0.0001, count=2000)

kf2 = SecondOrderKF(R, 0, dt=1)
fxs2, ps2 = filter_data(kf2, zs2000)

plot_kf_output(xs2000, fxs2, zs2000)
plot_residuals(xs2000[:,0], fxs2[:,0], ps2[:,0],
               'Seceond Order Position Residuals',
               'meters')
</pre>

Again, the residual plot tells the story. The track looks very good, but the residual plot shows that the filter is diverging for significant periods of time.

How should you think about all of this? You might argue that the last plot is 'good enough' for your application, and perhaps it is. I warn you however that a diverging filter doesn't always converge. With a different data set, or a physical system that performs differently you can end up with a filter that never converges.

Also, let's think about this in a data fitting sense. Suppose I give you two points, and tell you to fit a straight line to the points.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plt.scatter([1,2], [1,1],s=100,c='r')
plt.plot([0,3], [1,1])
plt.show()
</pre>

There is only one possible answer, to draw a line between the two points. Furthermore, the answer is optimal. If I gave you more points you could use a least squares fit to find the best line, and the answer would still be optimal (in a least squares sense).

But suppose I told you to fit a higher order polynomial to those two points. There is now an infinite number of answers to the problem. For example, for a second order system any parabola that passes through those two points would 'fit' the data, and there are an infinite number of parabolas that do that. When the Kalman filter is of higher order than your physical process it also has an infinite number of solutions to choose from. The answer is not just non-optimal, it often diverges and never reacquires the signal.

So this is the story of Goldilocks. You don't want a filter too big, or too small, but one that is just right. In many cases that will be easy to do - if you are designing a Kalman filter to read the thermometer of a freezer it seems clear that a zero order filter is the right choice. But what order should we use if we are tracking a car? Order one will work well while the car is moving in a straight line at a constant speed, but cars turn, speed up, and slow down, in which case a second order filter will perform better. That is the problem addressed in the Adaptive Filters chapter. There we will learn how to design a filter that *adapts* to changing order in the tracked object's behavior.

With that said, a lower order filter can track a higher order process so long as you add enough process noise. The results will not be optimal, but they can still be very good, and I always reach for this tool first before trying an adaptive filter. Let's look at an example with acceleration. First, the simulation.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
class ConstantAccelerationObject(object):
    def __init__(self, x0=0, vel=1., acc=0.1, acc_noise_scale=.1):
        self.x = x0
        self.vel = vel
        self.acc = acc
        self.acc_noise_scale = acc_noise_scale

    def update(self):
        self.acc += randn()*self.acc_noise_scale
        self.vel += self.acc
        self.x += self.vel
        return (self.x, self.vel, self.acc)


R = 6.
Q = 0.02
def simulate_acc_system(R, Q, count):
    obj = ConstantAccelerationObject(acc_noise_scale=Q)
    zs = []
    xs = []
    for i in range(count):
        x = obj.update()
        z = sense(x, R)
        xs.append(x)
        zs.append(z)
    return np.asarray(xs), zs

np.random.seed(124)
xs,zs = simulate_acc_system(R=R, Q=Q, count=80)
plt.plot(xs[:,0])
plt.show()
</pre>

Now we will filter the data using a second order filter.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
kf2 = SecondOrderKF(R, Q, dt=1)
fxs2, ps2 = filter_data(kf2, zs)

plot_kf_output(xs, fxs2, zs)
plot_residuals(xs[:,0], fxs2[:,0], ps2[:,0],
               'Second Order Position Residuals',
               'meters')
</pre>

We can see that the filter is performing within the theoretical limits of the filter.

Now let's use a lower order filter. As already demonstrated the lower order filter will lag the signal because it is not modeling the acceleration. However, we can account for that (to an extent) by increasing the size of the process noise. The filter will treat the acceleration as noise in the process model. The result will be suboptimal, but if designed well it will not diverge. Choosing the amount of extra process noise is not an exact science. You will have to experiment with representative data. Here, I've multiplied it by 10, and am getting good results.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
kf2 = FirstOrderKF(R, Q*10, dt=1)
fxs2, ps2 = filter_data(kf2, zs)

plot_kf_output(xs, fxs2, zs)
plot_residuals(xs[:,0], fxs2[:,0], ps2[:,0],
               'Second Order Position Residuals',
               'meters')
</pre>

Think about what will happen if you make the process noise many times larger than it needs to be. A large process noise tells the filter to favor the measurements, so we would expect the filter to closely mimic the noise in the measurements. Let's find out.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
kf2 = FirstOrderKF(R, Q*10000, dt=1)
fxs2, ps2 = filter_data(kf2, zs)

plot_kf_output(xs, fxs2, zs)
plot_residuals(xs[:,0], fxs2[:,0], ps2[:,0],
               'Second Order Position Residuals',
               'meters')
</pre>

## Sensor Fusion

Early in the g-h filter chapter we discussed designing a filter for two scales, one that was accurate and one that was very inaccurate. We determined that we should always include the information from the inaccurate filter - we should never discard any information that is available to us. If you do not recall this discussion, consider this edge case. You have two scales, one accurate to 1 lb, the second to 10 lbs. You way yourself on each scale, and the first reads 170 lbs, and the second reads 181. If we only used the data from the first scale we would only know that our weight is in the range of 169 to 171 lbs. However, when we incorporate the measurement from the second scale we can decide that the correct weight can only be 171 lbs, since 171 lbs is exactly at the outside range of the second measurement, which would be 171 to 191 lbs. Of course most measurements don't offer such a exact answer, but the principle applies to any measurement.

So consider a situation where we have two sensors measuring the same value at each time. How shall we incorporate that into our Kalman filter?

Suppose we have a train or cart on a railway. It has a sensor attached to the wheels counting revolutions, which can be converted to a distance along the track. Then, suppose we have a GPS-like sensor which I'll call a 'PS' mounted to the train which reports position. Thus, we have two measurements, both reporting position along the track. Suppose further that the accuracy of the wheel sensor is 1m, and the accuracy of the PS is 10m. How do we combine these two measurements into one filter? This may seem quite contrived, but aircraft use sensor fusion to fuse the measurements from sensors such as a GPS, INS, Doppler radar, VOR, the airspeed indicator, and more.

Kalman filters for inertial filters are very difficult, but fusing data from two or more sensors providing measurements of the same state variable (such as position) is actually quite easy. The relevant matrix is the measurement matrix <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span>. Recall that this matrix tells us how to convert from the Kalman filter's state <span class="math-tex" data-type="tex">\\(\mathbf{X}\\)</span> to a measurement <span class="math-tex" data-type="tex">\\({Z}\\)</span>. Suppose that we decide that our Kalman filter state should contain the position and velocity of the train, so that

<span class="math-tex" data-type="tex">\\( \mathbf{X} = \begin{bmatrix}x \\ \dot{x}\end{bmatrix}\\)</span>

We have two measurements for position, so we will define the measurement vector to be a vector of the measurements from the wheel and the PS.

<span class="math-tex" data-type="tex">\\( \mathbf{Z} = \begin{bmatrix}x_{wheel} \\ x_{ps}\end{bmatrix}\\)</span>

So we have to design the matrix <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> to convert <span class="math-tex" data-type="tex">\\(\mathbf{X}\\)</span> to <span class="math-tex" data-type="tex">\\(\mathbf{Z}\\)</span> . They are both positions, so the conversion is nothing more than multiplying by one:

<span class="math-tex" data-type="tex">\\( \begin{bmatrix}x_{wheel} \\ x_{ps}\end{bmatrix} = \begin{bmatrix}1 &0 \\ 1& 0\end{bmatrix} \begin{bmatrix}x \\ \dot{x}\end{bmatrix}\\)</span>

To make it clearer, suppose that the wheel reports not position but the number of rotations of the wheels, where 1 revolution yields 2m of travel. In that case we would write

<span class="math-tex" data-type="tex">\\( \begin{bmatrix}x_{wheel} \\ x_{ps}\end{bmatrix} = \begin{bmatrix}0.5 &0 \\ 1& 0\end{bmatrix} \begin{bmatrix}x \\ \dot{x}\end{bmatrix}\\)</span>

Now we have to design the measurement noise matrix <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span>. Suppose that the measurement variance for the PS is twice the variance of the wheel, and the standard deviation of the wheel is 1.5m. That gives us

<span class="math-tex" data-type="tex">\\(\sigma_{wheel} =  1.5\\
\sigma^2_{wheel} = 2.25 \\
\sigma_{ps} = 1.5*2 = 3 \\
\sigma^2_{ps} = 9.
\\)</span>

That is pretty much our Kalman filter design. We need to design for Q, but that is invariant to whether we are doing sensor fusion or not, so I will just choose some arbitrary value for Q.

So let's run a simulation of this design. I will assume a velocity of 10m/s with an update rate of 0.1 seconds.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy.random as random
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from numpy import array, asarray

def fusion_test(wheel_sigma, ps_sigma, do_plot=True):
    dt = 0.1
    kf = KalmanFilter(dim_x=2, dim_z=2)

    kf.F = array([[1., dt], [0., 1.]])
    kf.H = array([[1., 0.], [1., 0.]])
    kf.x = array([[0.], [1.]])
    kf.Q *= array([[(dt**3)/3, (dt**2)/2],
                   [(dt**2)/2,  dt      ]]) * 0.02
    kf.P *= 100
    kf.R[0,0] = wheel_sigma**2
    kf.R[1,1] = ps_sigma**2

    random.seed(1123)
    xs = []
    zs = []
    nom = []
    for i in range(1, 100):
        m0 = i + randn()*wheel_sigma
        m1 = i + randn()*ps_sigma
        z = array([[m0], [m1]])

        kf.predict()
        kf.update(z)

        xs.append(kf.x.T[0])
        zs.append(z.T[0])
        nom.append(i)

    xs = asarray(xs)
    zs = asarray(zs)
    nom = asarray(nom)


    res = nom-xs[:,0]
    print('fusion std: {:.3f}'.format (np.std(res)))
    if do_plot:
        bp.plot_measurements(zs[:,0], label='Wheel')
        plt.plot(zs[:,1], linestyle='--', label='Pos Sensor')
        bp.plot_filter(xs[:,0], label='Kalman filter')
        plt.legend(loc=4)
        plt.ylim(0,100)
        plt.show()

fusion_test(1.5, 3.0)
</pre>

We can see the result of the Kalman filter in blue.

It may be somewhat difficult to understand the previous example at an intuitive level. Let's look at a different problem. Suppose we are tracking an object in 2D space, and have two radar systems at different positions. Each radar system gives us a range and bearing to the target. How do the readings from each data affect the results?

This is a nonlinear problem because we need to use a trigonometry  to compute coordinates from a range and bearing, and we have not yet learned how to solve nonlinear problems with Kalman filters. So for this problem ignore the code that I used and just concentrate on the charts that the code outputs. We will revisit this problem in subsequent chapters and learn how to write this code.

I will position the target at (100,100). The first radar will be at (50,50), and the second radar at (150, 50). This will cause the first radar to measure a bearing of 45 degrees, and the second will report 135 degrees.

I will create the Kalman filter first, and then plot its initial covariance matrix. I am using what is called an Unscented Kalman filter, which is the subject of a later chapter.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from filterpy.kalman import  UnscentedKalmanFilter as UKF

def hx(x):
    """ compute measurements corresponding to state x"""
    dx = x[0] - hx.radar_pos[0]
    dy = x[1] - hx.radar_pos[1]
    return np.array([atan2(dy,dx), (dx**2 + dy**2)**.5])

def fx(x,dt):
    """ predict state x at 'dt' time in the future"""
    return x

# create unscented Kalman filter with large initial uncertainty
kf = UKF(2, 2, dt=0.1, hx=hx, fx=fx, kappa=2.)
kf.x = np.array([100, 100.])
kf.P *= 40

x0 = kf.x.copy()
p0 = kf.P.copy()

stats.plot_covariance_ellipse(
       x0, cov=p0,
       facecolor='y', edgecolor=None, alpha=0.6)
</pre>

We are equally uncertain about the position in x and y, so the covariance is circular.

Now we will update the Kalman filter with a reading from the first radar. I will set the standard deviation of the bearing error at 0.5<span class="math-tex" data-type="tex">\\(^\circ\\)</span>, and the standard deviation of the distance error at 3.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from math import radians, atan2

# set the error of the radar's bearing and distance
kf.R[0,0] = radians (.5)**2
kf.R[1,1] = 3.**2

# compute position and covariance from first radar station
hx.radar_pos = (50, 50)
dist = (50**2 + 50**2) ** 0.5
kf.predict()
kf.update([radians(45), dist])


# plot the results
x1 = kf.x.copy()
p1 = kf.P.copy()

stats.plot_covariance_ellipse(
       x0, cov=p0,
       facecolor='y', edgecolor=None, alpha=0.6)

stats.plot_covariance_ellipse(
       x1, cov=p1,
       facecolor='g', edgecolor='k', alpha=0.6)

plt.scatter([100], [100], c='y', label='Initial')
plt.scatter([100], [100], c='g', label='1st station')
plt.legend(scatterpoints=1, markerscale=3)
plt.plot([92,100],[92,100], c='g', lw=2, ls='--')
plt.show()
</pre>

We can see the effect of the errors on the geometry of the problem. The radar station is to the lower left of the target. The bearing measurement is extremely accurate at <span class="math-tex" data-type="tex">\\(\sigma=0.5^\circ\\)</span>, but the distance error is inaccurate at <span class="math-tex" data-type="tex">\\(\sigma=3\\)</span>. I've shown the radar reading with the dotted green line. We can easily see the effect of the accurate bearing and inaccurate distance in the shape of the covariance ellipse.

Now we can incorporate the second radar station's measurement. The second radar is at (150,50), which is below and to the right of the target. Before you go on, think about how you think the covariance will change when we incorporate this new reading.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# compute position and covariance from first radar station
hx.radar_pos = (150, 50)
kf.predict()
kf.update([radians(135), dist])

stats.plot_covariance_ellipse(
       x0, cov=p0,
       facecolor='y', edgecolor='k', alpha=0.6)

stats.plot_covariance_ellipse(
       x1, cov=p1,
       facecolor='g', edgecolor='k', alpha=0.6)

stats.plot_covariance_ellipse(
       kf.x, cov=kf.P,
       facecolor='b', edgecolor='k', alpha=0.6)

plt.scatter([100], [100], c='y', label='Initial')
plt.scatter([100], [100], c='g', label='1st station')
plt.scatter([100], [100], c='b', label='2nd station')
plt.legend(scatterpoints=1, markerscale=3)
plt.plot([92,100],[92,100], c='g', lw=2, ls='--')
plt.plot([108,100],[92,100], c='b', lw=2, ls='--')
plt.show()
</pre>

We can see how the second radar measurement altered the covariance. The angle to the target is orthogonal to the first radar station, so the effects of the error in the bearing and range are swapped. So the angle of the covariance matrix switches to match the direction to the second station. It is important to note that the direction did not merely change; the size of the covariance matrix became much smaller as well.

The covariance will always incorporate all of the information available, including the effects of the geometry of the problem. This formulation makes it particularly easy to see what is happening, but the same thing occurs if one sensor gives you position and a second sensor gives you velocity, or if two sensors provide measurements of position.

### Exercise: Can you Kalman Filter GPS outputs?

In the section above I have you apply a Kalman filter to 'GPS-like' sensor. Can you apply a Kalman filter to the output of a commercial Kalman filter? In other words, will the output of your filter be better than, worse than, or equal to the GPS's output?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# Think your answer here
</pre>

#### Solution

Commercial GPS's have a Kalman filter built into them, and their output is the filtered estimate created by that filter. So, suppose you have a steady stream of output from the GPS consisting of a position and position error. Can you not pass those two pieces of data into your own filter?

Well, what are the characteristics of that data stream, and more importantly, what are the fundamental requirements of the input to the Kalman filter?

Inputs to the Kalman filter must be *Gaussian* and *time independent*. The output of the GPS is *time dependent* because the filter bases it's current estimate on the recursive estimates of all previous measurements. Hence, the signal is not white, it is not time independent, and if you pass that data into a Kalman filter you have violated the mathematical requirements of the filter. So, the answer is no, you cannot get better estimates by running a KF on the output of a commercial GPS.

Another way to think of it is that Kalman filters are optimal in a least squares sense. There is no way to take an optimal solution, pass it through a filter, any filter, and get a 'more optimal' answer because it is a logical impossibility. At best the signal will be unchanged, in which case it will still be optimal, or it will be changed, and hence no longer optimal.

This is a difficult problem that hobbyists face when trying to integrate GPS, IMU's and other off the shelf sensors. It is too early in the book to try to address this problem, hence the 'PS' in the this section.

But let's look at the effect. A commercial GPS reports position, and an estimated error range. The estimated error just comes from the Kalman filter's <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> matrix. So let's filter some noisy data, take the filtered output as the new noisy input to the filter, and see what the result is. In other words, <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> will supply the  <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span> input, and  <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> will supply the measurement covariance <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span>. To exaggerate the effects somewhat to make them more obvious I will plot the effects of doing this one, and then a second time. The second iteration doesn't make any 'sense' (no one would try that), it just helps me illustrate a point. First, the code and plots.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
np.random.seed(124)
xs,zs = simulate_acc_system(R=R, Q=Q, count=80)

# Filter measurements
kf0 = SecondOrderKF(R, Q, dt=1)
fxs0, ps0 = filter_data(kf0, zs)


# filter again, using P as error
kf1 = SecondOrderKF(R, Q, dt=1)
fxs1 = []
ps1 = []

for z, p in zip(fxs0[:,0], ps0):
    kf1.predict()
    kf1.update(z, R=p[0])
    fxs1.append(kf1.x)
    ps1.append(kf1.P)

# one more time to exagerrate the effect
fxs1 = np.asarray(fxs1)
ps1  = np.asarray(ps1)

kf2 = SecondOrderKF(R, Q, dt=1)
fxs2 = []
for z, p in zip(fxs1[:,0], ps1):
    kf2.predict()
    kf2.update(z, R=p[0,0])
    fxs2.append(kf2.x)

fxs2 = np.asarray(fxs2)

plot_kf_output(xs[0:30], fxs0[0:30], zs[0:30], title='KF')
plot_kf_output(xs[0:30], fxs1[0:30], zs[0:30], title='1 iteration')
plot_kf_output(xs[0:30], fxs2[0:30], zs[0:30], title='2 iterations')
</pre>

We see that the filtered output of the reprocessed signal is smoother, but it also diverges from the track. What is happening? Recall that the Kalman filter requires that the signal not be time correlated. However the output of the Kalman filter *is* time correlated because it incorporates all previous measurements into its estimate for this time period. So look at the last graph, for 2 iterations. The measurements start with several peaks that are larger than the track. This is 'remembered' (that is vague terminology, but I am trying to avoid the math) by the filter, and it has started to compute that the object is above the track. Later, at around 13 seconds we have a period where the measurements all happen to be below the track. This also gets incorporated into the memory of the filter, and the iterated output diverges far below the track.

Now let's look at this in a different way. The iterated output is *not* using <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span> as the measurement, but the output of the previous Kalman filter estimate. So I will plot the output of the filter against the previous filter's output.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plot_kf_output(xs[0:30], fxs0[0:30], zs[0:30], title='KF')
plot_kf_output(xs[0:30], fxs1[0:30], fxs0[0:30:,0], title='1 iteration')
plot_kf_output(xs[0:30], fxs2[0:30], fxs1[0:30:,0], title='2 iterations')
</pre>

I hope the problem with this approach is now apparent. In the bottom graph we can see that the KF is tracking the imperfect estimates of the previous filter, and incorporating delay into the signal as well due to the memory of the previous measurements being incorporated into the signal.

### Excercise: Prove that the PS improves the filter

Devise a way to prove that fusing the PS and wheel measurements yields a better result than using the wheel alone.

#### Solution 1

Force the Kalman filter to disregard the PS measurement by setting the measurement noise for the PS to a near infinite value. Re-run the filter and observe the standard deviation of the residual.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
fusion_test(1.5, 3.0, do_plot=False)
fusion_test(1.5, 1e80, do_plot=False)
</pre>

Here we can see the error in the filter where the PS measurement is almost entirely ignored is greater than that where it is used.

#### Solution 2

This is more work, but we can write a Kalman filter that only takes one measurement.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy.random as random
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from numpy import array, asarray

dt = 0.1
wheel_sigma = 1.5
kf = KalmanFilter(dim_x=2, dim_z=1)

kf.F = array ([[1., dt], [0., 1.]])
kf.H = array ([[1., 0.]])
kf.x = array ([[0.], [1.]])
kf.Q *= 0.01
kf.P *= 100
kf.R[0,0] = wheel_sigma**2

random.seed(1123)
xs = []
zs = []
nom = []
for i in range(1, 100):
    m0 = i + randn()*wheel_sigma
    z = array([[m0]])

    kf.predict()
    kf.update(z)

    xs.append(kf.x.T[0])
    zs.append(z.T[0])
    nom.append(i)

xs = asarray(xs)
zs = asarray(zs)
nom = asarray(nom)


res = nom-xs[:,0]
print('std: {:.3f}'.format (np.std(res)))

bp.plot_filter(xs[:,0], label='Kalman filter')
bp.plot_measurements(zs[:,0],label='Wheel')
plt.legend(loc=4)
plt.xlim((-1,100))
plt.show()
</pre>

On this run I got a standard deviation of 0.523 vs the value of 0.391 for the fused measurements.

### Exercise: Different Data Rates

It is rare that two different sensor classes output data at the same rate. Assume that the PS produces an update at 1 Hz, and the wheel updates at 4 Hz. Design a filter that incorporates all of these measurements.

**hint**: This is a difficult problem in that I have not explained how to do this. Think about which matrices incorporate time, and which incorporate knowledge about the number and kind of measurements. All of these will have to be designed to work with this problem. If you can correctly enumerate those matrices you are most of the way to a solution.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
#your solution here!
</pre>

#### Solution

We can do this by setting the data rate to 0.25 seconds, which is 4 Hz. As we loop, on every iteration we call `update()` with only the wheel measurement. Then, every fourth time we will call `update()` with both the wheel and PS measurements.

This means that we vary the amount of data in the z parameter. The matrices associated with the measurement are <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> and <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span>. In the code above we designed H to be

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}1 &0 \\ 1& 0\end{bmatrix}\\)</span>

to account for the two measurements of position. When only the wheel reading is available, we must set
<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}1 &0\end{bmatrix}.\\)</span>

The matrix <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span> specifies the noise in each measurement. In the code above we set

<span class="math-tex" data-type="tex">\\(\mathbf{R} = \begin{bmatrix}\sigma_{wheel}^2 &0 \\ 0 & \sigma_{PS}^2\end{bmatrix}\\)</span>

When only the wheel measurement is available, we must set

<span class="math-tex" data-type="tex">\\(\mathbf{R} = \begin{bmatrix}\sigma_{wheel}^2\end{bmatrix}\\)</span>

The two matrices that incorporate time are <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> and <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span>. For example,

<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}1 & \Delta t \\ 0& 1\end{bmatrix}.\\)</span>

Since the wheel and PS reading coincide once every 4 readings we can just set <span class="math-tex" data-type="tex">\\(\delta t =0.25\\)</span> and not modify it while filtering. If the readings did not coincide in each iteration you would have to calculate how much time has passed since the last predict, compute a new <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> and <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span>, and then call `predict()` so the filter could make a correct prediction based on the time step required.

So here is the code.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy.random as random
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from numpy import array, asarray

def fusion_test(wheel_sigma, ps_sigma, do_plot=True):
    dt = 0.25
    kf = KalmanFilter(dim_x=2, dim_z=2)

    kf.F = array([[1., dt], [0., 1.]])
    kf.H = array([[1., 0.], [1., 0.]])
    kf.x = array([[0.], [1.]])
    kf.Q *= array([[(dt**3)/3, (dt**2)/2],
                   [(dt**2)/2,  dt      ]]) * 0.02
    kf.P *= 100
    kf.R[0,0] = wheel_sigma**2
    kf.R[1,1] = ps_sigma**2

    random.seed(1123)
    xs = []
    zs_wheel = []
    zs_ps = []
    nom = []
    for i in range(1, 101):

        if i % 4 == 0:
            m0 = i + randn()*wheel_sigma
            m1 = i + randn()*ps_sigma
            z = array([[m0], [m1]])
            kf.H = array([[1., 0.], [1., 0.]])
            R = np.eye(2)
            R[0,0] = wheel_sigma**2
            R[1,1] = ps_sigma**2

            zs_wheel.append(m0)
            zs_ps.append((i,m1))
        else:
            m0 = i + randn()*wheel_sigma
            z = array([m0])
            kf.H = array([[1., 0.]])
            R = np.eye(1) * wheel_sigma**2
            zs_wheel.append(m0)

        kf.predict()
        kf.update(z, R)

        xs.append(kf.x.T[0])
        nom.append(i)

    xs = asarray(xs)
    nom = asarray(nom)

    res = nom-xs[:,0]
    print('fusion std: {:.3f}'.format (np.std(res)))
    if do_plot:
        bp.plot_measurements(zs_wheel,  label='Wheel')
        plt.plot(*zip(*zs_ps), linestyle='--', label='Pos Sensor')
        bp.plot_filter(xs[:,0], label='Kalman filter')
        plt.legend(loc=4)
        plt.ylim(0,100)
        plt.show()

fusion_test(1.5, 3.0)
</pre>

We can see from the standard deviation that the performance is a bit worse than when the PS and wheel were measured in every update, but better than the wheel alone.

The code is fairly straightforward. The `update()` function optionally takes R as an argument, and I chose to do that rather than alter `KalmanFilter.R`, mostly to show that it is possible. Either way is fine. I modified `KalmanFilter.H` on each update depending on whether there are 1 or 2 measurements available. The only other difficulty was storing the wheel and PS measurements in two different arrays because there are a different number of measurements for each.

## Tracking a Ball

Now let's turn our attention to a situation where the physics of the object that we are tracking is constrained. A ball thrown in a vacuum must obey Newtonian laws. In a constant gravitational field it will travel in a parabola. I will assume you are familiar with the derivation of the formula:

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
y &= \frac{g}{2}t^2 + v_{y0} t + y_0 \\
x &= v_{x0} t + x_0
\end{aligned}
\\)</span>

where <span class="math-tex" data-type="tex">\\(g\\)</span> is the gravitational constant, <span class="math-tex" data-type="tex">\\(t\\)</span> is time, <span class="math-tex" data-type="tex">\\(v_{x0}\\)</span> and <span class="math-tex" data-type="tex">\\(v_{y0}\\)</span> are the initial velocities in the x and y plane. If the ball is thrown with an initial velocity of <span class="math-tex" data-type="tex">\\(v\\)</span> at angle <span class="math-tex" data-type="tex">\\(\theta\\)</span> above the horizon, we can compute <span class="math-tex" data-type="tex">\\(v_{x0}\\)</span> and <span class="math-tex" data-type="tex">\\(v_{y0}\\)</span> as

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
v_{x0} = v \cos{\theta} \\
v_{y0} = v \sin{\theta}
\end{aligned}
\\)</span>

Because we don't have real data we will start by writing a simulator for a ball. As always, we add a noise term independent of time so we can simulate noise sensors.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from math import radians, sin, cos
import math

def rk4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.
    y is the initial value for y
    x is the initial value for x
    dx is the difference in x (e.g. the time step)
    f is a callable function (y, x) that you supply to
      compute dy/dx for the specified values.
    """

    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.

def fx(x,t):
    return fx.vel

def fy(y,t):
    return fy.vel - 9.8*t


class BallTrajectory2D(object):
    def __init__(self, x0, y0, velocity,
                 theta_deg=0.,
                 g=9.8,
                 noise=[0.0,0.0]):
        self.x = x0
        self.y = y0
        self.t = 0

        theta = math.radians(theta_deg)

        fx.vel = math.cos(theta) * velocity
        fy.vel = math.sin(theta) * velocity

        self.g = g
        self.noise = noise


    def step(self, dt):
        self.x = rk4(self.x, self.t, dt, fx)
        self.y = rk4(self.y, self.t, dt, fy)
        self.t += dt
        return (self.x + random.randn()*self.noise[0],
                self.y + random.randn()*self.noise[1])
</pre>

So to create a trajectory starting at (0,15) with a velocity of <span class="math-tex" data-type="tex">\\(100 \frac{m}{s}\\)</span> and an angle of <span class="math-tex" data-type="tex">\\(60^\circ\\)</span> we would write:

    traj = BallTrajectory2D (x0=0, y0=15, velocity=100, theta_deg=60)

and then call `traj.position(t)` for each time step. Let's test this

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def test_ball_vacuum(noise):
    y = 15
    x = 0
    ball = BallTrajectory2D(x0=x, y0=y,
                            theta_deg=60., velocity=100.,
                            noise=noise)
    t = 0
    dt = 0.25
    while y >= 0:
        x,y = ball.step(dt)
        t += dt
        if y >= 0:
            plt.scatter(x,y)

    plt.axis('equal')
    plt.show()

test_ball_vacuum([0,0]) # plot ideal ball position
test_ball_vacuum([1,1]) # plot with noise
</pre>

This looks reasonable, so let's continue (exercise for the reader: validate this simulation more robustly).

### Step 1: Choose the State Variables

We might think to use the same state variables as used for tracking the dog. However, this will not work. Recall that the Kalman filter state transition must be written as <span class="math-tex" data-type="tex">\\(\mathbf{x}' = \mathbf{F x}\\)</span>, which means we must calculate the current state from the previous state. Our assumption is that the ball is traveling in a vacuum, so the velocity in x is a constant, and the acceleration in y is solely due to the gravitational constant <span class="math-tex" data-type="tex">\\(g\\)</span>. We can discretize the Newtonian equations using the well known Euler method in terms of <span class="math-tex" data-type="tex">\\(\Delta t\\)</span> are:

<span class="math-tex" data-type="tex">\\(\begin{aligned}
x_t &=  x_{t-1} + v_{x(t-1)} {\Delta t} \\
v_{xt} &= vx_{t-1}
\\
y_t &= -\frac{g}{2} {\Delta t}^2 + vy_{t-1} {\Delta t} + y_{t-1} \\
v_{yt} &= -g {\Delta t} + v_{y(t-1)} \\
\end{aligned}
\\)</span>
> **sidebar**: *Euler's method integrates a differential equation stepwise by assuming the slope (derivative) is constant at time <span class="math-tex" data-type="tex">\\(t\\)</span>. In this case the derivative of the position is velocity. At each time step <span class="math-tex" data-type="tex">\\(\Delta t\\)</span> we assume a constant velocity, compute the new position, and then update the velocity for the next time step. There are more accurate methods, such as Runge-Kutta available to us, but because we are updating the state with a measurement in each step Euler's method is very accurate.*

This implies that we need to incorporate acceleration for <span class="math-tex" data-type="tex">\\(y\\)</span> into the Kalman filter, but not for <span class="math-tex" data-type="tex">\\(x\\)</span>. This suggests the following state variables.

<span class="math-tex" data-type="tex">\\(
\mathbf{x} =
\begin{bmatrix}
x \\
\dot{x} \\
y \\
\dot{y} \\
\ddot{y}
\end{bmatrix}
\\)</span>

### **Step 2:**  Design State Transition Function

Our next step is to design the state transition function. Recall that the state transistion function is implemented as a matrix <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> that we multiply with the previous state of our system to get the next state<span class="math-tex" data-type="tex">\\(\mathbf{x}' = \mathbf{Fx}\\)</span>.

I will not belabor this as it is very similar to the 1-D case we did in the previous chapter. Our state equations for position and velocity would be:

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
x' &= (1*x) + (\Delta t * v_x) + (0*y) + (0 * v_y) + (0 * a_y) \\
v_x &= (0*x) +  (1*v_x) + (0*y) + (0 * v_y) + (0 * a_y) \\
y' &= (0*x) + (0* v_x)         + (1*y) + (\Delta t * v_y) + (\frac{1}{2}{\Delta t}^2*a_y)  \\
v_y &= (0*x) +  (0*v_x) + (0*y) + (1*v_y) + (\Delta t * a_y) \\
a_y &= (0*x) +  (0*v_x) + (0*y) + (0*v_y) + (1 * a_y)
\end{aligned}
\\)</span>

Note that none of the terms include <span class="math-tex" data-type="tex">\\(g\\)</span>, the gravitational constant. This is because the state variable <span class="math-tex" data-type="tex">\\(\ddot{y}\\)</span> will be initialized with <span class="math-tex" data-type="tex">\\(g\\)</span>, or -9.81. Thus the function <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> will propagate <span class="math-tex" data-type="tex">\\(g\\)</span> through the equations correctly.

In matrix form we write this as:

<span class="math-tex" data-type="tex">\\(
\mathbf{F} = \begin{bmatrix}
1 & \Delta t & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & \Delta t & \frac{1}{2}{\Delta t}^2 \\
0 & 0 & 0 & 1 & \Delta t \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
\\)</span>

### Interlude: Test State Transition

The Kalman filter class provides us with useful defaults for all of the class variables, so let's take advantage of that and test the state transition function before continuing. Here we construct a filter as specified in Step 2 above. We compute the initial velocity in x and y using trigonometry, and then set the initial condition for <span class="math-tex" data-type="tex">\\(x\\)</span>.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from math import sin,cos,radians

def ball_kf(x, y, omega, v0, dt):

    g = 9.8 # gravitational constant

    kf = KalmanFilter(dim_x=5, dim_z=2)


    ay = .5*dt**2

    kf.F = np.array([[1, dt,  0,  0,  0],   # x   = x0+dx*dt
                     [0,  1,  0,  0,  0],   # dx  = dx
                     [0,  0,  1, dt, ay],   # y   = y0 +dy*dt+1/2*g*dt^2
                     [0,  0,  0,  1, dt],   # dy  = dy0 + ddy*dt
                     [0,  0,  0,  0, 1]])   # ddy = -g.

    # compute velocity in x and y
    omega = radians(omega)
    vx = cos(omega) * v0
    vy = sin(omega) * v0

    kf.Q *= 0.
    kf.x = np.array([[x, vx, y, vy, -g]]).T

    return kf
</pre>

Now we can test the filter by calling predict until <span class="math-tex" data-type="tex">\\(y=0\\)</span>, which corresponds to the ball hitting the ground. We will graph the output against the idealized computation of the ball's position. If the model is correct, the Kalman filter prediction should match the ideal model very closely. We will draw the ideal position with a green circle, and the Kalman filter's output with '+' marks.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
y = 15.
x = 0.
theta = 20. # launch angle
v0 = 100.
dt = 0.1   # time step

ball = BallTrajectory2D(x0=x, y0=y, theta_deg=theta,
                        velocity=v0, noise=[0,0])
kf = ball_kf(x, y, theta, v0, dt)
t = 0
while kf.x[2, 0] > 0:
    t += dt
    kf.predict()
    x, y = ball.step(dt)
    p2 = plt.scatter(x, y,
                     color='g', marker='o', s=75,
                     alpha=0.5)
    p1 = plt.scatter(kf.x[0, 0], kf.x[2, 0],
                     color='k', marker='+', s=200)

plt.legend([p1, p2], ['Kalman filter', 'idealized'], scatterpoints=1)
plt.show()
</pre>

As we can see, the Kalman filter agrees with the physics model very closely. If you are interested in pursuing this further, try altering the initial velocity, the size of dt, and <span class="math-tex" data-type="tex">\\(\theta\\)</span>, and plot the error at each step. However, the important point is to test your design as soon as possible; if the design of the state transition is wrong all subsequent effort may be wasted. More importantly, it can be extremely difficult to tease out an error in the state transition function when the filter incorporates measurement updates.

### **Step 3**: Design the Motion Function

We have no control inputs to the ball flight, so this step is trivial - set the motion transition function <span class="math-tex" data-type="tex">\\(\small\mathbf{B}=0\\)</span>. This is done for us by the class when it is created so we can skip this step.

### **Step 4**: Design the Measurement Function

The measurement function defines how we go from the state variables to the measurements using the equation <span class="math-tex" data-type="tex">\\(\mathbf{z} = \mathbf{Hx}\\)</span>. We will assume that we have a sensor that provides us with the position of the ball in (x,y), but cannot measure velocities or accelerations. Therefore our function must be:

<span class="math-tex" data-type="tex">\\(
\begin{bmatrix}z_x \\ z_y \end{bmatrix}=
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0
\end{bmatrix} *
\begin{bmatrix}
x \\
\dot{x} \\
y \\
\dot{y} \\
\ddot{y}\end{bmatrix}\\)</span>

where

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0
\end{bmatrix}\\)</span>

### **Step 5**: Design the Measurement Noise Matrix

As with the robot, we will assume that the error is independent in <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span>. In this case we will start by assuming that the measurement error in x and y are 0.5 meters. Hence,

<span class="math-tex" data-type="tex">\\(\mathbf{R} = \begin{bmatrix}0.5&0\\0&0.5\end{bmatrix}\\)</span>

### Step 6: Design the Process Noise Matrix

Finally, we design the process noise. As with the robot tracking example, we don't yet have a good way to model process noise. However, we are assuming a ball moving in a vacuum, so there should be no process noise. For now we will assume the process noise is 0 for each state variable. This is a bit silly - if we were in a perfect vacuum then our predictions would be perfect, and we would have no need for a Kalman filter. We will soon alter this example to be more realistic by incorporating air drag and ball spin.

We have 5 state variables, so we need a <span class="math-tex" data-type="tex">\\(5{\times}5\\)</span> covariance matrix:

<span class="math-tex" data-type="tex">\\(\mathbf{Q} = \begin{bmatrix}0&0&0&0&0\\0&0&0&0&0\\0&0&0&0&0\\0&0&0&0&0\\0&0&0&0&0\end{bmatrix}\\)</span>

### Step 7: Design the Initial Conditions

We already performed this step when we tested the state transition function. Recall that we computed the initial velocity for <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> using trigonometry, and set the value of <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> with:

    omega = radians(omega)
    vx = cos(omega) * v0
    vy = sin(omega) * v0

    f1.x = np.array([[x, vx, y, vy, -g]]).T


With all the steps done we are ready to implement our filter and test it. First, the implementation:

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from math import sin,cos,radians

def ball_kf(x, y, omega, v0, dt, r=0.5, q=0.):

    g = 9.8 # gravitational constant
    kf = KalmanFilter(dim_x=5, dim_z=2)

    ay = .5*dt**2
    kf.F = np.array ([[1, dt,  0,  0,  0],   # x   = x0+dx*dt
                      [0,  1,  0,  0,  0],   # dx  = dx
                      [0,  0,  1, dt, ay],   # y   = y0 +dy*dt+1/2*g*dt^2
                      [0,  0,  0,  1, dt],   # dy  = dy0 + ddy*dt
                      [0,  0,  0,  0, 1]])   # ddy = -g.

    kf.H = np.array([[1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]])

    kf.R *= r
    kf.Q *= q

    omega = radians(omega)
    vx = cos(omega) * v0
    vy = sin(omega) * v0

    kf.x = np.array([[x, vx, y, vy, -9.8]]).T

    return kf
</pre>

Now we will test the filter by generating measurements for the ball using the ball simulation class.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
y = 1.
x = 0.
theta = 35. # launch angle
v0 = 80.
dt = 1/10.   # time step

ball = BallTrajectory2D(x0=x, y0=y, theta_deg=theta, velocity=v0,
                        noise=[.2,.2])
kf = ball_kf(x,y,theta,v0,dt)

t = 0
xs = []
ys = []
while kf.x[2,0] > 0:
    t += dt
    x,y = ball.step(dt)
    z = np.array([[x, y]]).T

    kf.update(z)
    xs.append(kf.x[0,0])
    ys.append(kf.x[2,0])

    kf.predict()

    p1 = plt.scatter(x, y, color='r', marker='.', s=75, alpha=0.5)

p2, = plt.plot (xs, ys,lw=2)
plt.legend([p2, p1], ['Kalman filter', 'Measurements'], scatterpoints=1)
plt.show()
</pre>

We see that the Kalman filter reasonably tracks the ball. However, as already explained, this is a silly example; we can predict trajectories in a vacuum with arbitrary precision; using a Kalman filter in this example is a needless complication.

## Tracking a Ball in Air

**author's note - I originally had ball tracking code in 2 different places in the book. One has been copied here, so now we have 2 sections on ball tracking. I need to edit this into one section, obviously. Sorry for the duplication.**

We are now ready to design a practical Kalman filter application. For this problem we assume that we are tracking a ball traveling through the Earth's atmosphere. The path of the ball is influenced by wind, drag, and the rotation of the ball. We will assume that our sensor is a camera; code that we will not implement will perform some type of image processing to detect the position of the ball. This is typically called *blob detection* in computer vision. However, image processing code is not perfect; in any given frame it is possible to either detect no blob or to detect spurious blobs that do not correspond to the ball. Finally, we will not assume that we know the starting position, angle, or rotation of the ball; the tracking code will have to initiate tracking based on the measurements that are provided. The main simplification that we are making here is a 2D world; we assume that the ball is always traveling orthogonal to the plane of the camera's sensor. We have to make that simplification at this point because we have not yet discussed how we might extract 3D information from a camera, which necessarily provides only 2D data.

### Implementing Air Drag

Our first step is to implement the math for a ball moving through air. There are several treatments available. A robust solution takes into account issues such as ball roughness (which affects drag non-linearly depending on velocity), the Magnus effect (spin causes one side of the ball to have higher velocity relative to the air vs the opposite side, so the coefficient of drag differs on opposite sides), the effect of lift, humidity, air density, and so on. I assume the reader is not interested in the details of ball physics, and so will restrict this treatment to the effect of air drag on a non-spinning baseball. I will use the math developed by Nicholas Giordano and Hisao Nakanishi in *Computational Physics*  [1997].

**Important**: Before I continue, let me point out that you will not have to understand this next piece of physics to proceed with the Kalman filter. My goal is to create a reasonably accurate behavior of a baseball in the real world, so that we can test how our Kalman filter performs with real-world behavior. In real world applications it is usually impossible to completely model the physics of a real world system, and we make do with a process model that incorporates the large scale behaviors. We then tune the measurement noise and process noise until the filter works well with our data. There is a real risk to this; it is easy to finely tune a Kalman filter so it works perfectly with your test data, but performs badly when presented with slightly different data. This is perhaps the hardest part of designing a Kalman filter, and why it gets referred to with terms such as 'black art'.

I dislike books that implement things without explanation, so I will now develop the physics for a ball moving through air. Move on past the implementation of the simulation if you are not interested.

A ball moving through air encounters wind resistance. This imparts a force on the wall, called *drag*, which alters the flight of the ball. In Giordano this is denoted as

<span class="math-tex" data-type="tex">\\(F_{drag} = -B_2v^2\\)</span>

where <span class="math-tex" data-type="tex">\\(B_2\\)</span> is a coefficient derived experimentally, and <span class="math-tex" data-type="tex">\\(v\\)</span> is the velocity of the object. <span class="math-tex" data-type="tex">\\(F_{drag}\\)</span> can be factored into <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> components with

<span class="math-tex" data-type="tex">\\(F_{drag,x} = -B_2v v_x\\
F_{drag,y} = -B_2v v_y
\\)</span>

If <span class="math-tex" data-type="tex">\\(m\\)</span> is the mass of the ball, we can use <span class="math-tex" data-type="tex">\\(F=ma\\)</span> to compute the acceleration as

<span class="math-tex" data-type="tex">\\( a_x = -\frac{B_2}{m}v v_x\\
a_y = -\frac{B_2}{m}v v_y\\)</span>

Giordano provides the following function for <span class="math-tex" data-type="tex">\\(\frac{B_2}{m}\\)</span>, which takes air density, the cross section of a baseball, and its roughness into account. Understand that this is an approximation based on wind tunnel tests and several simplifying assumptions. It is in SI units: velocity is in meters/sec and time is in seconds.

<span class="math-tex" data-type="tex">\\(\frac{B_2}{m} = 0.0039 + \frac{0.0058}{1+\exp{[(v-35)/5]}}\\)</span>

Starting with this Euler discretization of the ball path in a vacuum:
<span class="math-tex" data-type="tex">\\(\begin{aligned}
x &= v_x \Delta t \\
y &= v_y \Delta t \\
v_x &= v_x \\
v_y &= v_y - 9.8 \Delta t
\end{aligned}
\\)</span>

We can incorporate this force (acceleration) into our equations by incorporating <span class="math-tex" data-type="tex">\\(accel * \Delta t\\)</span> into the velocity update equations. We should subtract this component because drag will reduce the velocity. The code to do this is quite straightforward, we just need to break out the Force into <span class="math-tex" data-type="tex">\\(x\\)</span> and <span class="math-tex" data-type="tex">\\(y\\)</span> components.

I will not belabor this issue further because the computational physics is beyond the scope of this book. Recognize that a higher fidelity simulation would require incorporating things like altitude, temperature, ball spin, and several other factors. My intent here is to impart some real-world behavior into our simulation to test how  our simpler prediction model used by the Kalman filter reacts to this behavior. Your process model will never exactly model what happens in the world, and a large factor in designing a good Kalman filter is carefully testing how it performs against real world data.

The code below computes the behavior of a baseball in air, at sea level, in the presence of wind. I plot the same initial hit with no wind, and then with a tail wind at 10 mph. Baseball statistics are universally done in US units, and we will follow suit here (http://en.wikipedia.org/wiki/United_States_customary_units). Note that the velocity of 110 mph is a typical exit speed for a baseball for a home run hit.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from math import sqrt, exp, cos, sin, radians

def mph_to_mps(x):
    return x * .447

def drag_force(velocity):
    """ Returns the force on a baseball due to air drag at
    the specified velocity. Units are SI"""

    return (0.0039 + 0.0058 / (1. + exp((velocity-35.)/5.))) * velocity

v = mph_to_mps(110.)
y = 1
x = 0
dt = .1
theta = radians(35)

def solve(x, y, vel, v_wind, launch_angle):
    xs = []
    ys = []
    v_x = vel*cos(launch_angle)
    v_y = vel*sin(launch_angle)
    while y >= 0:
        # Euler equations for x and y
        x += v_x*dt
        y += v_y*dt

        # force due to air drag
        velocity = sqrt ((v_x-v_wind)**2 + v_y**2)
        F = drag_force(velocity)

        # euler's equations for vx and vy
        v_x = v_x - F*(v_x-v_wind)*dt
        v_y = v_y - 9.8*dt - F*v_y*dt

        xs.append(x)
        ys.append(y)

    return xs, ys

x,y = solve(x=0, y=1, vel=v, v_wind=0, launch_angle=theta)
p1 = plt.scatter(x, y, color='blue', label='no wind')

x,y = solve(x=0, y=1,vel=v, v_wind=mph_to_mps(10), launch_angle=theta)
p2 = plt.scatter(x, y, color='green', marker="v", label='10mph wind')
plt.legend(scatterpoints=1)
plt.show()
</pre>

We can easily see the difference between the trajectory in a vacuum and in the air. I used the same initial velocity and launch angle in the ball in a vacuum section above. We computed that the ball in a vacuum would travel over 240 meters (nearly 800 ft). In the air, the distance is just over 120 meters, or roughly 400 ft. 400ft is a realistic distance for a well hit home run ball, so we can be confident that our simulation is reasonably accurate.

Without further ado we will create a ball simulation that uses the math above to create a more realistic ball trajectory. I will note that the nonlinear behavior of drag means that there is no analytic solution to the ball position at any point in time, so we need to compute the position step-wise. I use Euler's method to propagate the solution; use of a more accurate technique such as Runge-Kutta is left as an exercise for the reader. That modest complication is unnecessary for what we are doing because the accuracy difference between the techniques will be small for the time steps we will be using.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from math import radians, sin, cos, sqrt, exp

class BaseballPath(object):
    def __init__(self, x0, y0, launch_angle_deg, velocity_ms,
                 noise=(1.0,1.0)):
        """ Create 2D baseball path object
           (x = distance from start point in ground plane,
            y=height above ground)

        x0,y0            initial position
        launch_angle_deg angle ball is travelling respective to
                         ground plane
        velocity_ms      speeed of ball in meters/second
        noise            amount of noise to add to each position
                         in (x,y)
        """

        omega = radians(launch_angle_deg)
        self.v_x = velocity_ms * cos(omega)
        self.v_y = velocity_ms * sin(omega)

        self.x = x0
        self.y = y0

        self.noise = noise


    def drag_force(self, velocity):
        """ Returns the force on a baseball due to air drag at
        the specified velocity. Units are SI
        """
        B_m = 0.0039 + 0.0058 / (1. + exp((velocity-35.)/5.))
        return B_m * velocity


    def update(self, dt, vel_wind=0.):
        """ compute the ball position based on the specified time
        step and wind velocity. Returns (x,y) position tuple.
        """

        # Euler equations for x and y
        self.x += self.v_x*dt
        self.y += self.v_y*dt

        # force due to air drag
        v_x_wind = self.v_x - vel_wind
        v = sqrt(v_x_wind**2 + self.v_y**2)
        F = self.drag_force(v)

        # Euler's equations for velocity
        self.v_x = self.v_x - F*v_x_wind*dt
        self.v_y = self.v_y - 9.81*dt - F*self.v_y*dt

        return (self.x + random.randn()*self.noise[0],
                self.y + random.randn()*self.noise[1])
</pre>

Now we can test the Kalman filter against measurements created by this model.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
y = 1.
x = 0.
theta = 35. # launch angle
v0 = 50.
dt = 1/10.   # time step

ball = BaseballPath(x0=x, y0=y, launch_angle_deg=theta,
                    velocity_ms=v0, noise=[.3,.3])
f1 = ball_kf(x,y,theta,v0,dt,r=1.)
f2 = ball_kf(x,y,theta,v0,dt,r=10.)
t = 0
xs = []
ys = []
xs2 = []
ys2 = []

while f1.x[2,0] > 0:
    t += dt
    x,y = ball.update(dt)
    z = np.array([[x, y]]).T

    f1.update(z)
    f2.update(z)
    xs.append(f1.x[0,0])
    ys.append(f1.x[2,0])
    xs2.append(f2.x[0,0])
    ys2.append(f2.x[2,0])
    f1.predict()
    f2.predict()

    p1 = plt.scatter(x, y, color='r', marker='o', s=75, alpha=0.5)

p2, = plt.plot (xs, ys, lw=2)
p3, = plt.plot (xs2, ys2, lw=4)
plt.legend([p1,p2, p3],
           ['Measurements', 'Kalman filter(R=0.5)', 'Kalman filter(R=10)'],
           loc='best', scatterpoints=1)
plt.show()
</pre>

I have plotted the output of two different Kalman filter settings. The measurements are depicted as green circles, a Kalman filter with R=0.5 as a thin blue line, and a Kalman filter with R=10 as a thick red line. These R values are chosen merely to show the effect of measurement noise on the output, they are not intended to imply a correct design.

We can see that neither filter does very well. At first both track the measurements well, but as time continues they both diverge. This happens because the state model for air drag is nonlinear and the Kalman filter assumes that it is linear. If you recall our discussion about nonlinearity in the g-h filter chapter we showed why a g-h filter will always lag behind the acceleration of the system. We see the same thing here - the acceleration is negative, so the Kalman filter consistently overshoots the ball position. There is no way for the filter to catch up so long as the acceleration continues, so the filter will continue to diverge.

What can we do to improve this? The best approach is to perform the filtering with a nonlinear Kalman filter, and we will do this in subsequent chapters. However, there is also what I will call an 'engineering' solution to this problem as well. Our Kalman filter assumes that the ball is in a vacuum, and thus that there is no process noise. However, since the ball is in air the atmosphere imparts a force on the ball. We can think of this force as process noise. This is not a particularly rigorous thought; for one thing, this force is anything but Gaussian. Secondly, we can compute this force, so throwing our hands up and saying 'it's random' will not lead to an optimal solution. But let's see what happens if we follow this line of thought.

The following code implements the same Kalman filter as before, but with a non-zero process noise. I plot two examples, one with `Q=.1`, and one with `Q=0.01`.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def plot_ball_with_q(q, r=1., noise=0.3):
    y = 1.
    x = 0.
    theta = 35. # launch angle
    v0 = 50.
    dt = 1/10.   # time step

    ball = BaseballPath(x0=x,
                        y0=y,
                        launch_angle_deg=theta,
                        velocity_ms=v0,
                        noise=[noise,noise])
    f1 = ball_kf(x,y,theta,v0,dt,r=r, q=q)
    t = 0
    xs = []
    ys = []

    while f1.x[2,0] > 0:
        t += dt
        x,y = ball.update(dt)
        z = np.array([[x,y]]).T

        f1.update(z)
        xs.append(f1.x[0,0])
        ys.append(f1.x[2,0])
        f1.predict()


        p1 = plt.scatter(x, y, color='r', marker='o', s=75, alpha=0.5)

    p2, = plt.plot (xs, ys,lw=2)
    plt.legend([p1, p2], ['Measurements', 'Kalman filter'], scatterpoints=1)
    plt.show()

plot_ball_with_q(0.01)
plot_ball_with_q(0.1)
</pre>

The second filter tracks the measurements fairly well. There appears to be a bit of lag, but very little.

Is this a good technique? Usually not, but it depends. Here the nonlinearity of the force on the ball is fairly constant and regular. Assume we are trying to track an automobile - the accelerations will vary as the car changes speeds and turns. When we make the process noise higher than the actual noise in the system the filter will opt to weigh the measurements higher. If you don't have a lot of noise in your measurements this might work for you. However, consider this next plot where I have increased the noise in the measurements.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plot_ball_with_q(0.01, r=3, noise=3.)
plot_ball_with_q(0.1, r=3, noise=3.)
</pre>

This output is terrible. The filter has no choice but to give more weight to the measurements than the process (prediction step), but when the measurements are noisy the filter output will just track the noise. This inherent limitation of the linear Kalman filter is what lead to the development of nonlinear versions of the filter.

With that said, it is certainly possible to use the process noise to deal with small nonlinearities in your system. This is part of the 'black art' of Kalman filters. Our model of the sensors and of the system are never perfect. Sensors are non-Gaussian and our process model is never perfect. You can mask some of this by setting the measurement errors and process errors higher than their theoretically correct values, but the trade off is a non-optimal solution. Certainly it is better to be non-optimal than to have your Kalman filter diverge. However, as we can see in the graphs above, it is easy for the output of the filter to be very bad. It is also very common to run many simulations and tests and to end up with a filter that performs very well under those conditions. Then, when you use the filter on real data the conditions are slightly different and the filter ends up performing terribly.

For now we will set this problem aside, as we are clearly misapplying the Kalman filter in this example. We will revisit this problem in subsequent chapters to see the effect of using various nonlinear techniques. In some domains you will be able to get away with using a linear Kalman filter for a nonlinear problem, but usually you will have to use one or more of the techniques you will learn in the rest of this book.

## Tracking Noisy Data

If we are applying a Kalman filter to a thermometer in an oven in a factory then our task is done once the Kalman filter is designed. The data from the thermometer may be noisy, but there is never doubt that the thermometer is reading the temperature of *some other* oven. Contrast this to our current situation, where we are using computer vision to detect ball blobs from a video camera. For any frame we may detect or may not detect the ball, and we may have one or more spurious blobs - blobs not associated with the ball at all. This can occur because of limitations of the computer vision code, or due to foreign objects in the scene, such as a bird flying through the frame. Also, in the general case we may have no idea where the ball starts from. A ball may be picked up, carried, and thrown from any position, for example. A ball may be launched within view of the camera, or the initial launch might be off screen and the ball merely travels through the scene. There is the possibility of bounces and deflections - the ball can hit the ground and bounce, it can bounce off a wall, a person, or any other object in the scene.

Consider some of the problems that can occur. We could be waiting for a ball to appear, and a blob is detected. We initialize our Kalman filter with that blob, and look at the next frame to detect where the ball is going. Maybe there is no blob in the next frame. Can we conclude that the blob in the previous frame was noise? Or perhaps the blob was valid, but we did not detect the blob in this frame.

**author's note: not sure if I want to cover this. If I do, not sure I want to cover this here.**

[Table of Contents](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

# Kalman Filter Math

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
#format the book
%matplotlib inline
%load_ext autoreload
%autoreload 2
from __future__ import division, print_function
import matplotlib.pyplot as plt
import book_format
book_format.load_style()
</pre>

** author's note:** *the ordering of material in this chapter is questionable. I delve into solving ODEs before discussing the basic Kalman equations. If you are reading this while it is being worked on (so long as this notice exists), you may find it easier to skip around a bit until I organize it better.*


If you've gotten this far I hope that you are thinking that the Kalman filter's fearsome reputation is somewhat undeserved. Sure, I hand waved some equations away, but I hope implementation has been fairly straightforward for you. The underlying concept is quite straightforward - take two measurements, or a measurement and a prediction, and choose the output to be somewhere between the two. If you believe the measurement more your guess will be closer to the measurement, and if you believe the prediction is more accurate your guess will lie closer it it. That's not rocket science (little joke - it is exactly this math that got Apollo to the moon and back!).

Well, to be honest I have been choosing my problems carefully. For any arbitrary problem finding some of the matrices that we need to feed into the Kalman filter equations can be quite difficult. I haven't been *too tricky*, though. Equations like Newton's equations of motion can be trivially computed for Kalman filter applications, and they make up the bulk of the kind of problems that we want to solve. If you are a hobbyist, you can safely pass by this chapter for now, and perhaps forever. Some of the later chapters will assume the material in this chapter, but much of the work will still be accessible to you.

But, I urge everyone to at least read the first section, and to skim the rest. It is not much harder than what you have done - the difficulty comes in finding closed form expressions for specific problems, not understanding the math in this chapter.

## Bayesian Probability

The title of this book is *Kalman and Bayesian Filters in Python* but to date I have not touched on the Bayesian aspect much. There was enough going on in the earlier chapters that adding this form of reasoning about filters could be a distraction rather than a help. I now which to take some time to explain what Bayesian probability is and how a Kalman filter is in fact a Bayesian filter. This is not just a diversion. First of all, a lot of the Kalman filter literature uses this formulation when talking about filters, so you will need to understand what they are talking about. Second, this math plays a strong role in filtering design once we move past the Kalman filter.

To do so we will go back to our first tracking problem - tracking a dog in a hallway. Recall the update step - we believed with some degree of precision that the dog was at position 7 (for example), and then receive a measurement that the dog is at position 7.5. We want to incorporate that measurement into our belief. In the *Discrete Bayes* chapter we used histograms to denote our estimates at each hallway position, and in the *One Dimensional Kalman Filters* we used Gaussians. Both are method of using *Bayesian* probability.

Briefly, *Bayesian* probability is a branch of math that lets us evaluate a hypothesis or new data point given some uncertain information about the past. For example, suppose you are driving down your neighborhood street and see one of your neighbors at their door, using a key to let themselves in. Three doors down you see two people in masks breaking a window of another house. What might you conclude?

It is likely that you would reason that in the first case your neighbors were getting home and unlocking their door to get inside. In the second case you at least strongly suspect a robbery is in progress. In the first case you would just proceed on, and in the second case you'd probably call the police.

Of course, things are not always what they appear. Perhaps unbeknownst to you your neighbor sold their house that morning, and they were now breaking in to steal the new owner's belongings. In the second case, perhaps the owners of the house were at a costume event at the next house, they had a medical emergency with their child, realized they lost their keys, and were breaking into their own house to get the much needed medication. Those are both *unlikely* events, but possible. Adding a few additional pieces of information would allow you to determine the true state of affairs in all but the most complicated situations.

These are instances of *Bayesian* reasoning. We take knowledge from the past and integrate in new information. You know that your neighbor owned their house yesterday, so it is still highly likely that they still own it today. You know that owners of houses normally have keys to the front door, and that the normal mode of entrance into your own house is not breaking windows, so the second case is *likely* to be a breaking and entering. The reasoning is not ironclad as shown by the alternative explanations, but it is likely.

### Bayes' theorem

*Bayes' theorem* mathematically formalizes the above reasoning. It is written as

<span class="math-tex" data-type="tex">\\(P(A|B) = \frac{P(B | A)\, P(A)}{P(B)}\cdot\\)</span>


Before we do some computations, let's review what the terms mean. P(A) is called the *prior probability* of the event A, and is often just shortened to the *prior*. What is the prior? It is just the probability of A being true *before* we incorporate new evidence. In our dog tracking problem above, the prior is the probability we assign to our belief that the dog is positioned at 7 before we make the measurement of 7.5. It is important to master this terminology if you expect to read a lot of the literature.

<span class="math-tex" data-type="tex">\\(P(A|B)\\)</span> is the *conditional probability* that A is true given that B is true. For example, if it is true that your neighbor still owns their house, then it will be very likely that they are not breaking into their house. In Bayesian probability this is called the *posterior*, and it denotes our new belief after incorporating the measurement/knowledge of B. For our dog tracking problem the posterior is the probability given to the estimated position after incorporating the measurement 7.5. For the neighbor problem the posterior would be the probability of a break in after you find out that your neighbor sold their home last week.

What math did we use for the dog tracking problem? Recall that we used this equation to compute the new mean and probability distribution

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
N(estimate) * N(measurement) &= \\
N(\mu_1, \sigma_1^2)*N(\mu_2, \sigma_2^2) &= N(\frac{\sigma_1^2 \mu_2 + \sigma_2^2 \mu_1}{\sigma_1^2 + \sigma_2^2},\frac{1}{\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}}) \cdot
\end{aligned}
\\)</span>


Here <span class="math-tex" data-type="tex">\\(N(\mu_1, \sigma_1^2)\\)</span> is the old estimated position, so <span class="math-tex" data-type="tex">\\(\sigma_1\\)</span> is an indication of our *prior* probability. <span class="math-tex" data-type="tex">\\(N(\mu_2, \sigma_2^2)\\)</span> is the mean and variance of our measurement, and so the result can be thought of as the new position and probability distribution after incorporating the new measurement. In other words, our *posterior distribution* is

<span class="math-tex" data-type="tex">\\(\frac{1}{{\sigma_{estimate}}^2} + \frac{1}{{\sigma_{measurement}}^2}\\)</span>

This is still a little hard to compare to Bayes' equation because we are dealing with probability distributions rather than probabilities. So let's cast our minds back to the discrete Bayes chapter where we computed the probability that our dog was at any given position in the hallway. It looked like this:

    def update(pos_belief, measure, p_hit, p_miss):
        for i in range(len(hallway)):
            if hallway[i] == measure:
                pos_belief[i] *= p_hit
            else:
                pos_belief[i] *= p_miss

        pos_belief /= sum(pos_belief)

Let's rewrite this using our newly learned terminology.

    def update(prior_probability, measure, prob_hit, prob_miss):
        posterior_probability = np.zeros(len(prior_probability))
        for i in range(len(hallway)):
            if hallway[i] == measure:
                posterior_probability[i] = prior_probability[i] * p_hit
            else:
                posterior_probability[i] = prior_probability[i] * p_miss

        return posterior_probability / sum(posterior_probability)


So what is this doing? It's multiplying the old belief that the dog is at position *i* (prior probability) with the probability that the measurement is correct for that position, and then dividing by the total probability for that new event.

Now let's look at Bayes' equation again.

<span class="math-tex" data-type="tex">\\(P(A|B) = \frac{P(B | A)\, P(A)}{P(B)}\cdot\\)</span>

It's the same thing being calculated by the code. Multiply the prior (<span class="math-tex" data-type="tex">\\(P(A)\\)</span>) by the probability of the measurement at each position (<span class="math-tex" data-type="tex">\\(P(B|A)\\)</span>) and divide by the total probability for the event (<span class="math-tex" data-type="tex">\\(P(B)\\)</span>).

In other words the first half of the Discrete Bayes chapter developed Bayes' equation from a thought experiment. I could have just presented Bayes' equation and then given you the Python routine above to implement it, but chances are you would not have understood *why* Bayes' equation works. Presenting the equation first is the normal approach of Kalman filtering texts, and I always found it extremely nonintuitive.

## Modeling a Dynamic System that Has Noise

We need to start by understanding the underlying equations and assumptions that the Kalman filter uses. We are trying to model real world phenomena, so what do we have to consider?

First, each physical system has a process. For example, a car traveling at a certain velocity goes so far in a fixed amount of time, and its velocity varies as a function of its acceleration. We describe that behavior with the well known Newtonian equations we learned in high school.


<span class="math-tex" data-type="tex">\\(
\begin{aligned}
v&=at\\
x &= \frac{1}{2}at^2 + v_0t + d_0
\end{aligned}
\\)</span>

And once we learned calculus we saw them in this form:

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
 \mathbf{v} &= \frac{d \mathbf{x}}{d t}\\
 \quad \mathbf{a} &= \frac{d \mathbf{v}}{d t}\\
 &= \frac{d^2 \mathbf{x}}{d t^2} \,\!
\end{aligned}
 \\)</span>

A typical problem would have you compute the distance traveled given a constant velocity or acceleration. But, of course we know this is not all that is happening. First, we do not have perfect measures of things like the velocity and acceleration - there is always noise in the measurements, and we have to model that. Second, no car travels on a perfect road. There are bumps that cause the car to slow down, there is wind drag, there are hills that raise and lower the speed. If we do not have explicit knowledge of these factors we lump them all together under the term "process noise".

Trying to model all of those factors explicitly and exactly is impossible for anything but the most trivial problem. I could try to include equations for things like bumps in the road, the behavior of the car's suspension system, even the effects of hitting bugs with the windshield, but the job would never be done - there would always be more effects to add and limits to our knowledge (how many bugs do we hit in an hour, for example). What is worse, each of those models would in themselves be a simplification - do I assume the wind is constant, that the drag of the car is the same for all angles of the wind, that the suspension act as perfect springs, that the suspension for each wheel acts identically, and so on.

So control theory makes a mathematically correct simplification. We acknowledge that there are many factors that influence the system that we either do not know or that we don't want to have to model. At any time <span class="math-tex" data-type="tex">\\(t\\)</span> we say that the actual value (say, the position of our car) is the predicted value plus some unknown process noise:

<span class="math-tex" data-type="tex">\\(
x(t) = x_{pred}(t) + noise(t)
\\)</span>

This is not meant to imply that <span class="math-tex" data-type="tex">\\(noise(t)\\)</span> is a function that we can derive analytically or that it is well behaved. If there is a bump in the road at <span class="math-tex" data-type="tex">\\(t=10\\)</span> then the noise factor will just incorporate that effect. Again, this is not implying that we model, compute, or even know the value of *noise(t)*, it is merely a statement of fact - we can *always* describe the actual value as the predicted value from our idealized model plus some other value.

Let's express this with linear algebra. Using the same notation from previous chapters, we can say that our model of the system (without noise) is:

<span class="math-tex" data-type="tex">\\( f(\mathbf{x}) = \mathbf{Fx}\\)</span>

That is, we have a set of linear equations that describe our system. For our car,
<span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> will be the coefficients for Newton's equations of motion.

Now we need to model the noise. We will just call that *w*, and add it to the equation.

<span class="math-tex" data-type="tex">\\( f(\mathbf{x}) = \mathbf{Fx} + \mathbf{w}\\)</span>

Finally, we need to consider inputs into the system. We are dealing with linear problems here, so we will assume that there is some input <span class="math-tex" data-type="tex">\\(u\\)</span> into the system, and that we have some linear model that defines how that input changes the system. For example, if you press down on the accelerator in your car the car will accelerate. We will need a matrix <span class="math-tex" data-type="tex">\\(\mathbf{B}\\)</span> to convert <span class="math-tex" data-type="tex">\\(u\\)</span> into the effect on the system. We just add that into our equation:

<span class="math-tex" data-type="tex">\\( f(\mathbf{x}) = \mathbf{Fx} + \mathbf{Bu} + \mathbf{w}\\)</span>

And that's it. That is one of the equations that Kalman set out to solve, and he found a way to compute an optimal solution if we assume certain properties of <span class="math-tex" data-type="tex">\\(w\\)</span>.

However, we took advantage of something I left mostly unstated in the last chapter. We were able to provide a definition for <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> because we were able to take advantage of the exact solution that Newtonian equations provide us. However, if you have an engineering background you will realize what a small class of problems that covers. If you don't, I will explain it next, and provide you with several ways to compute <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span> for arbitrary systems.

## Modeling Dynamic Systems

Modeling dynamic systems is properly the topic of at least one undergraduate course in mathematics. To an extent there is no substitute for a few semesters of ordinary and partial differential equations. If you are a hobbyist, or trying to solve one very specific filtering problem at work you probably do not have the time and/or inclination to devote a year or more to that education.

However, I can present enough of the theory to allow us to create the system equations for many different Kalman filters, and give you enough background to at least follow the mathematics in the literature. My goal is to get you to the stage where you can read a Kalman filtering book or paper and understand it well enough to implement the algorithms. The background math is deep, but we end up using a few simple techniques over and over again in practice.

Let's lay out the problem and discuss what the solution will be. We  model *dynamic systems* with a set of first order *differential equations*. This should not be a surprise as calculus is the math of of thing that vary. For example, we say that velocity is the derivative of distance with respect to time

<span class="math-tex" data-type="tex">\\(\mathbf{v}= \frac{d \mathbf{x}}{d t} = \dot{\mathbf{x}}\\)</span>

where <span class="math-tex" data-type="tex">\\(\dot{\mathbf{x}}\\)</span> is the notation for the derivative of <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> with respect to t.

We need to use these equations for the predict step of the Kalman filter. Given the state of the system at time <span class="math-tex" data-type="tex">\\(t\\)</span> we want to predict its state at time <span class="math-tex" data-type="tex">\\(t + \Delta t\\)</span>. The Kalman filter matrices do not accept differential equations, so we need a mathematical technique that will find the solution to those equations at each time step. In general it is extremely difficult to find analytic solutions to systems of differential equations, so we will normally use *numerical* techniques to find accurate approximations for these equations.

### Why This is Hard

We model dynamic systems with a set of first order differential equations. For example, we already presented the Newtonian equation

<span class="math-tex" data-type="tex">\\(\mathbf{v}=\dot{\mathbf{x}}\\)</span>

where <span class="math-tex" data-type="tex">\\(\dot{\mathbf{x}}\\)</span> is the notation for the derivative of <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> with respect to t, or <span class="math-tex" data-type="tex">\\(\frac{d \mathbf{x}}{d t}\\)</span>.

In general terms we can then say that a dynamic system consists of equations of the form

<span class="math-tex" data-type="tex">\\( g(t) = \dot{x}\\)</span>

if the behavior of the system depends on time. However, if the system is *time invariant* the equations are of the form
<span class="math-tex" data-type="tex">\\( f(x) = \dot{x}\\)</span>

What does *time invariant* mean? Consider a home stereo. If you input a signal <span class="math-tex" data-type="tex">\\(x\\)</span> into it at time <span class="math-tex" data-type="tex">\\(t\\)</span>, it will output some signal <span class="math-tex" data-type="tex">\\(f(x)\\)</span>. If you instead make the input at a later time <span class="math-tex" data-type="tex">\\(t + \Delta t\\)</span> the output signal will still be exactly the same, just shifted in time. This is different from, say, an aircraft. If you make a control input to the aircraft at a later time it's behavior will be different because it will have burned additional fuel (and thus lost weight), drag may be different due to being at a different altitude, and so on.

We can solve these equations by integrating each side. The time variant equation is very straightforward. We essentially solved this problem with the Newtonian equations above, but let's be explicit and write it out. Starting with  <span class="math-tex" data-type="tex">\\(\dot{\mathbf{x}}=\mathbf{v}\\)</span> we get the expected

<span class="math-tex" data-type="tex">\\( \int \dot{\mathbf{x}}\mathrm{d}t = \int \mathbf{v} \mathrm{d}t\\
x = vt + x_0\\)</span>

However, integrating the time invariant equation is not so straightforward.

<span class="math-tex" data-type="tex">\\( \dot{x} = f(x) \\
\frac{dx}{dt} = f(x)
\\)</span>

Using the *separation of variables* techniques, we divide by <span class="math-tex" data-type="tex">\\(f(x)\\)</span> and move the <span class="math-tex" data-type="tex">\\(dx\\)</span> term to the right so we can integrate each side:

<span class="math-tex" data-type="tex">\\(
\int^x_{x_0} \frac{1}{f(x)} dx = \int^t_{t_0} dt\\
\\)</span>

If we let the solution to the left hand side by named <span class="math-tex" data-type="tex">\\(F(x)\\)</span>, we get

<span class="math-tex" data-type="tex">\\(F(x) - f(x_0) = t-t_0\\)</span>

We then solve for x with

<span class="math-tex" data-type="tex">\\(F(x) = t - t_0 + F(x_0) \\
x = F^{-1}[t-t_0 + F(x_0)]\\)</span>

In other words, we need to find the inverse of <span class="math-tex" data-type="tex">\\(F\\)</span>. This is not at all trivial, and a significant amount of course work in a STEM education is devoted to finding tricky, analytic solutions to this problem, backed by several centuries of research.

In the end, however, they are tricks, and many simple forms of <span class="math-tex" data-type="tex">\\(f(x)\\)</span> either have no closed form solution, or pose extreme difficulties. Instead, the practicing engineer turns to numerical methods to find a solution to her problems. I would suggest that students would be better served by learning fewer analytic mathematical tricks and instead focusing on learning numerical methods, but that is the topic for another book.

### Finding the Fundamental Matrix for Time Invariant Systems

 If you already have the mathematical training in solving partial differential equations you may be able to put it to use; I am not assuming that sort of background.  So let me skip over quite a bit of mathematics and present the typical numerical techniques used in Kalman filter design.

First, we express the system equations in state-space form (i.e. using linear algebra equations) with

<span class="math-tex" data-type="tex">\\( \dot{\mathbf{x}} = \mathbf{Fx}\\)</span>

Now we can assert that we want to find the fundamental matrix <span class="math-tex" data-type="tex">\\(\Phi\\)</span> that propagates the state with the equation

<span class="math-tex" data-type="tex">\\(x(t) = \Phi(t-t_0)x(t_0)\\)</span>

In other words, we just want to compute the value of <span class="math-tex" data-type="tex">\\(x\\)</span> at time <span class="math-tex" data-type="tex">\\(t\\)</span> by multiplying its previous value by some matrix <span class="math-tex" data-type="tex">\\(\Phi\\)</span>. This is not trivial to do because the original equations do not include time

Broadly speaking there are three ways to find <span class="math-tex" data-type="tex">\\(\Phi\\)</span>. The technique most often used with Kalman filters is to use a Taylor-series expansion:

<span class="math-tex" data-type="tex">\\( \Phi(t) = e^{\mathbf{F}t} = \mathbf{I} + \mathbf{F}t  + \frac{(\mathbf{F}t)^2}{2!} + \frac{(\mathbf{F}t)^3}{3!} + ... \\)</span>

This is much easy to compute, and thus is the typical approach used in Kalman filter design when the filter is reasonably small. If you are wondering where <span class="math-tex" data-type="tex">\\(e\\)</span> came from, I point you to the Wikipedia article on the matrix exponential [1]. Here the important point is to recognize the very simple and regular form this equation takes. We will put this form into use in the next chapter, so I will not belabor its use here.

*Linear Time Invariant Theory*, also known as LTI System Theory, gives us a way to find <span class="math-tex" data-type="tex">\\(\Phi\\)</span> using the inverse Laplace transform. You are either nodding your head now, or completely lost. Don't worry, I will not be using the Laplace transform in this book except in this paragraph, as the computation is quite difficult to perform in practice. LTI system theory tells us that

<span class="math-tex" data-type="tex">\\( \Phi(t) = \mathcal{L}^{-1}[(s\mathbf{I} - \mathbf{F})^{-1}]\\)</span>

I have no intention of going into this other than to say that the inverse Laplace transform converts a signal into the frequency (time) domain, but finding a solution to the equation above is non-trivial. If you are interested, the Wikipedia article on LTI system theory provides an introduction [2].

Finally, there are numerical techniques to find <span class="math-tex" data-type="tex">\\(\Phi\\)</span>. As filters get larger finding analytical solutions becomes very tedious (though packages like SymPy make it easier). C. F. van Loan [3] has developed a technique that finds both <span class="math-tex" data-type="tex">\\(\Phi\\)</span> and <span class="math-tex" data-type="tex">\\(Q\\)</span> numerically.

I have implemented van Loan's method in `FilterPy`. You may use it as follows:

    from filterpy.common import van_loan_discretization

    F = np.array([[0,1],[-1,0]], dtype=float)
    G = np.array([[0.],[2.]]) # white noise scaling
    phi, Q = van_loan_discretization(F, G, dt=0.1)

See the docstring documentation for van_loan_discretization for more information, which I have embedded below.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from filterpy.common import van_loan_discretization
help(van_loan_discretization)
</pre>

### Forming First Order Equations from Higher Order Equations

In the sections above I spoke of *first order* differential equations; these are equations with only first derivatives. However, physical systems often require second or higher order equations. Any higher order system of equations can be converted to a first order set of equations by defining extra variables for the first order terms and then solving. Let's do an example.

Given the system <span class="math-tex" data-type="tex">\\(\ddot{x} - 6\dot{x} + 9x = t\\)</span> find the first order equations.


The first step is to isolate the highest order term onto one side of the equation .

<span class="math-tex" data-type="tex">\\(\ddot{x} = 6\dot{x} - 9x + t\\)</span>

We define two new variables:

<span class="math-tex" data-type="tex">\\( x_1(t) = x \\
x_2(t) = \dot{x}
\\)</span>

Now we will substitute these into the original equation and solve, giving us a set of first order equations in terms of these new variables.

First, we know that <span class="math-tex" data-type="tex">\\(\dot{x}_1 = x_2\\)</span> and that <span class="math-tex" data-type="tex">\\(\dot{x}_2 = \ddot{x}\\)</span>. Therefore

<span class="math-tex" data-type="tex">\\(\begin{aligned}
\dot{x}_2 &= \ddot{x} \\
          &= 6\dot{x} - 9x + t\\
          &= 6x_2-9x_1 + t
\end{aligned}\\)</span>

Therefore our first order system of equations is

<span class="math-tex" data-type="tex">\\(\begin{aligned}\dot{x}_1 &= x_2 \\
\dot{x}_2 &= 6x_2-9x_1 + t\end{aligned}\\)</span>

If you practice this a bit you will become adept at it. Just isolate the highest term, define a new variable and its derivatives, and then substitute.

## Walking Through the Kalman Filter Equations

I promised that you would not have to understand how to derive Kalman filter equations, and that is true. However, I do think it is worth walking through the equations one by one and becoming familiar with the variables. If this is your first time through the material feel free to skip ahead to the next section. However, you will eventually want to work through this material, so why not now? You will need to have passing familiarity with these equations to read material written about the Kalman filter, as they all presuppose that you are familiar with them. I will reiterate them here for easy reference.


<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\text{Predict Step}\\
\mathbf{x} &= \mathbf{F x} + \mathbf{B u}\;\;\;\;&(1) \\
\mathbf{P} &= \mathbf{FP{F}}^\mathsf{T} + \mathbf{Q}\;\;\;\;&(2) \\
\\
\text{Update Step}\\
\textbf{y} &= \mathbf{z} - \mathbf{H}\mathbf{x}\;\;\;&(3) \\
\mathbf{S} &= \mathbf{HPH}^\mathsf{T} + \mathbf{R} \;\;\;&(4) \\
\mathbf{K} &= \mathbf{PH}^\mathsf{T}\mathbf{S}^{-1}\;\;\;&(5) \\
\mathbf{x} &= \mathbf{x} +\mathbf{K}\mathbf{y} \;\;\;&(6)\\
\mathbf{P} &= (\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}\;\;\;&(7)
\end{aligned}
\\)</span>

I will start with the update step, as that is what we started with in the one dimensional Kalman filter case. The first equation is

<span class="math-tex" data-type="tex">\\(
\mathbf{y} = \mathbf{z} - \mathbf{H x}\tag{3}
\\)</span>

On the right we have <span class="math-tex" data-type="tex">\\(\mathbf{Hx}\\)</span>. That should be recognizable as the measurement function. Multiplying <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> with <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> puts <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> into *measurement space*; in other words, the same basis and units as the sensor's measurements. The variable <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span> is just the measurement; it is typical, but not universal to use <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span> to denote measurements in the literature (<span class="math-tex" data-type="tex">\\(\mathbf{y}\\)</span> is also sometimes used). Do you remember this chart?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import mkf_internal
mkf_internal.show_residual_chart()
</pre>

The blue point labeled "prediction" is the output of <span class="math-tex" data-type="tex">\\(\mathbf{Hx}\\)</span>, and the dot labeled "measurement" is <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span>. Therefore, <span class="math-tex" data-type="tex">\\(\mathbf{y} = \mathbf{z} - \mathbf{Hx}\\)</span> is how we compute the residual, drawn in red, where <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span> is the residual.

The next two lines are the formidable:

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\mathbf{S} &= \textbf{HPH}^\mathsf{T} + \textbf{R} \;\;\;&(4) \\
\textbf{K} &= \textbf{PH}^\mathsf{T}\mathbf{S}^{-1}\;\;\;&(5) \\
\end{aligned}
\\)</span>
Unfortunately it is a fair amount of linear algebra to derive this. The derivation can be quite elegant, and I urge you to look it up if you have the mathematical education to follow it. But <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> is just the *Kalman gain* - the ratio of how much measurement vs prediction we should use to create the new estimate. <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span> is the *measurement noise*, and <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> is our *uncertainty covariance matrix* from the prediction step.

So let's work through this expression by expression. Start with <span class="math-tex" data-type="tex">\\(\mathbf{HPH}^\mathsf{T}\\)</span>. The linear equation <span class="math-tex" data-type="tex">\\(\mathbf{ABA}^T\\)</span> can be thought of as changing the basis of <span class="math-tex" data-type="tex">\\(\mathbf{B}\\)</span> to <span class="math-tex" data-type="tex">\\(\mathbf{A}\\)</span>. So <span class="math-tex" data-type="tex">\\(\mathbf{HPH}^\mathsf{T}\\)</span> is taking the covariance <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> and putting it in measurement (<span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span>) space.

In English, consider the problem of reading a temperature with a thermometer that provides readings in volts. Our state is in terms of temperature, but we are now doing calculations in *measurement space* - volts. So we need to convert <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> from applying to temperatures to volts. The linear algebra form <span class="math-tex" data-type="tex">\\(\textbf{H}\textbf{P}\textbf{H}^\mathsf{T}\\)</span> takes <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> to the basis used by <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span>, namely volts.

Then, once in measurement space, we can add the measurement noise <span class="math-tex" data-type="tex">\\(\mathbf{R}\\)</span> to it. Hence, the expression for the uncertainty once we include the measurement is:

<span class="math-tex" data-type="tex">\\(\mathbf{S} = \mathbf{HP}\mathbf{H}^\mathsf{T} + \mathbf{R}\\)</span>

The next equation is
<span class="math-tex" data-type="tex">\\(\textbf{K} = \textbf{P}\textbf{H}^T\mathbf{S}^{-1}\\
\\)</span>

<span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> is the *Kalman gain* - the ratio that chooses how far along the residual to select between the measurement and prediction in the graph above.

We can think of the inverse of a matrix as linear algebra's way of computing  <span class="math-tex" data-type="tex">\\(\frac{1}{x}\\)</span>. So we can read the equation for <span class="math-tex" data-type="tex">\\(\textbf{K}\\)</span> as

<span class="math-tex" data-type="tex">\\( \textbf{K} = \frac{\textbf{P}\textbf{H}^\mathsf{T}}{\mathbf{S}} \\)</span>


<span class="math-tex" data-type="tex">\\(
\textbf{K} = \frac{uncertainty_{prediction}}{uncertainty_{measurement}}\textbf{H}^\mathsf{T}
\\)</span>


In other words, the *Kalman gain* equation is doing nothing more than computing a ratio based on how much we trust the prediction vs the measurement. If we are confident in our measurements and unconfident in our predictions <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> will favor the measurement, and vice versa. The equation is complicated because we are doing this in multiple dimensions via matrices, but the concept is simple - scale by a ratio.

Without going into the derivation of <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span>, I'll say that this equation is the result of finding a value of <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> that optimizes the *mean-square estimation error*. It does this by finding the minimal values for <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> along its diagonal. Recall that the diagonal of <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> is just the variance for each state variable. So, this equation for <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> ensures that the Kalman filter output is optimal. To put this in concrete terms, for our dog tracking problem this means that the estimates for both position and velocity will be optimal - a value of <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> that made the position extremely accurate but the velocity very inaccurate would be rejected in favor of a <span class="math-tex" data-type="tex">\\(\mathbf{K}\\)</span> that made both position and velocity just somewhat accurate.

Our next line is:
 <span class="math-tex" data-type="tex">\\(\mathbf{x}=\mathbf{x}' +\mathbf{Ky}\tag{5}\\)</span>

This just multiplies the residual by the Kalman gain, and adds it to the state variable. In other words, this is the computation of our new estimate.

Finally, we have:

<span class="math-tex" data-type="tex">\\(\mathbf{P}=(\mathbf{I}-\mathbf{KH})\mathbf{P}\tag{6}\\)</span>

<span class="math-tex" data-type="tex">\\(I\\)</span> is the identity matrix, and is the way we represent <span class="math-tex" data-type="tex">\\(1\\)</span> in multiple dimensions. <span class="math-tex" data-type="tex">\\(H\\)</span> is our measurement function, and is a constant.  So, simplified, this is simply <span class="math-tex" data-type="tex">\\(P = (1-cK)P\\)</span>. <span class="math-tex" data-type="tex">\\(K\\)</span> is our ratio of how much prediction vs measurement we use. So, if <span class="math-tex" data-type="tex">\\(K\\)</span> is large then <span class="math-tex" data-type="tex">\\((1-cK)\\)</span> is small, and P will be made smaller than it was. If <span class="math-tex" data-type="tex">\\(K\\)</span> is small, then <span class="math-tex" data-type="tex">\\((1-cK)\\)</span> is large, and P will be made larger than it was. So we adjust the size of our uncertainty by some factor of the *Kalman gain*. I would like to draw your attention back to the g-h filter, which included this Python code:

    # update filter
    w = w * (1-scale_factor) + z * scale_factor

This multidimensional Kalman filter equation is partially implementing this calculation for the variance instead of the state variable.

Now we have the measurement steps. The first equation is

<span class="math-tex" data-type="tex">\\(\mathbf{x} = \mathbf{Fx} + \mathbf{Bu}\tag{1}\\)</span>

This is just our state transition equation which we have already discussed. <span class="math-tex" data-type="tex">\\(\mathbf{Fx}\\)</span>  multiplies <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span> with the state transition matrix to compute the next state. <span class="math-tex" data-type="tex">\\(B\\)</span> and <span class="math-tex" data-type="tex">\\(u\\)</span> add in the contribution of the control input <span class="math-tex" data-type="tex">\\(\mathbf{u}\\)</span>, if any.

The final equation is:
<span class="math-tex" data-type="tex">\\(\mathbf{P} = \mathbf{FPF}^\mathsf{T} + \mathbf{Q}\tag{2}\\)</span>

<span class="math-tex" data-type="tex">\\(\mathbf{FPF}^\mathsf{T}\\)</span> is the way we put <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span> into the process space using linear algebra so that we can add in the process noise <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> to it.

## Design of the Process Noise Matrix

In general the design of the <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> matrix is among the most difficult aspects of Kalman filter design. This is due to several factors. First, the math itself is somewhat difficult and requires a good foundation in signal theory. Second, we are trying to model the noise in something for which we have little information. For example, consider trying to model the process noise for a baseball. We can model it as a sphere moving through the air, but that leave many unknown factors - the wind, ball rotation and spin decay, the coefficient of friction of a scuffed ball with stitches, the effects of wind and air density, and so on. I will develop the equations for an exact mathematical solution for a given process model, but since the process model is incomplete the result for <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> will also be incomplete. This has a lot of ramifications for the behavior of the Kalman filter. If <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> is too small than the filter will be overconfident in it's prediction model and will diverge from the actual solution. If <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> is too large than the filter will be unduly influenced by the noise in the measurements and perform sub-optimally. In practice we spend a lot of time running simulations and evaluating collected data to try to select an appropriate value for <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span>. But let's start by looking at the math.


Let's assume a kinematic system - some system that can be modeled using Newton's equations of motion. We can make a few different assumptions about this process.

We have been using a process model of

<span class="math-tex" data-type="tex">\\( f(\mathbf{x}) = \mathbf{Fx} + \mathbf{w}\\)</span>

where <span class="math-tex" data-type="tex">\\(\mathbf{w}\\)</span> is the process noise. Kinematic systems are *continuous* - their inputs and outputs can vary at any arbitrary point in time. However, our Kalman filters are *discrete*. We sample the system at regular intervals. Therefore we must find the discrete representation for the noise term in the equation above. However, this depends on what assumptions we make about the behavior of the noise. We will consider two different models for the noise.

### Continuous White Noise Model

We model kinematic systems using Newton's equations. So far in this book we have either used position and velocity, or position,velocity, and acceleration as the models for our systems. There is nothing stopping us from going further - we can model jerk, jounce, snap, and so on. We don't do that normally because adding terms beyond the dynamics of the real system actually degrades the solution.

Let's say that we need to model the position, velocity, and acceleration. We can then assume that acceleration is constant. Of course, there is process noise in the system and so the acceleration is not actually constant. In this section we will assume that the acceleration changes by a continuous time zero-mean white noise <span class="math-tex" data-type="tex">\\(w(t)\\)</span>. In other words, we are assuming that velocity is acceleration changing by small amounts that over time average to 0 (zero-mean).


Since the noise is changing continuously we will need to integrate to get the discrete noise for the discretization interval that we have chosen. We will not prove it here, but the equation for the discretization of the noise is

<span class="math-tex" data-type="tex">\\(\mathbf{Q} = \int_0^{\Delta t} \Phi(t)\mathbf{Q_c}\Phi^\mathsf{T}(t) dt\\)</span>

where <span class="math-tex" data-type="tex">\\(\mathbf{Q_c}\\)</span> is the continuous noise. This gives us

<span class="math-tex" data-type="tex">\\(\Phi = \begin{bmatrix}1 & \Delta t & {\Delta t}^2/2 \\ 0 & 1 & \Delta t\\ 0& 0& 1\end{bmatrix}\\)</span>

for the fundamental matrix, and

<span class="math-tex" data-type="tex">\\(\mathbf{Q_c} = \begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix} \Phi_s\\)</span>

for the continuous process noise matrix, where <span class="math-tex" data-type="tex">\\(\Phi_s\\)</span> is the spectral density of the white noise.

We could carry out these computations ourselves, but I prefer using SymPy to solve the equation.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import sympy
from sympy import  init_printing, Matrix,MatMul, integrate, symbols

init_printing(use_latex='mathjax')
dt, phi = symbols('\Delta{t} \Phi_s')
F_k = Matrix([[1, dt, dt**2/2],
              [0,  1,      dt],
              [0,  0,       1]])
Q_c = Matrix([[0,0,0],
              [0,0,0],
              [0,0,1]])*phi

Q=sympy.integrate(F_k*Q_c*F_k.T,(dt, 0, dt))

# factor phi out of the matrix to make it more readable
Q = Q/phi
sympy.MatMul(Q, phi)
</pre>

For completeness, let us compute the equations for the 0th order and 1st order equations.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
F_k = sympy.Matrix([[1]])
Q_c = sympy.Matrix([[phi]])

print('0th order discrete process noise')
sympy.integrate(F_k*Q_c*F_k.T,(dt, 0, dt))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
F_k = sympy.Matrix([[1, dt],
                    [0, 1]])
Q_c = sympy.Matrix([[0,0],
                    [0,1]])*phi

Q = sympy.integrate(F_k*Q_c*F_k.T,(dt, 0, dt))

print('1st order discrete process noise')
# factor phi out of the matrix to make it more readable
Q = Q/phi
sympy.MatMul(Q, phi)
</pre>

### Piecewise White Noise Model

Another model for the noise assumes that the that highest order term (say, acceleration) is constant for each time period, but differs for each time period, and each of these is uncorrelated between time periods. This is subtly different than the model above, where we assumed that the last term had a continuously varying noisy signal applied to it.

We will model this as

<span class="math-tex" data-type="tex">\\(f(x)=Fx+\Gamma w\\)</span>

where <span class="math-tex" data-type="tex">\\(\Gamma\\)</span> is the *noise gain* of the system, and <span class="math-tex" data-type="tex">\\(w\\)</span> is the constant piecewise acceleration (or velocity, or jerk, etc).


Lets start with by looking a first order system. In this case we have the state transition function

<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}1&\Delta t \\ 0& 1\end{bmatrix}\\)</span>

In one time period, the change in velocity will be <span class="math-tex" data-type="tex">\\(w(t)\Delta t\\)</span>, and the change in position will be <span class="math-tex" data-type="tex">\\(w(t)\Delta t^2/2\\)</span>, giving us

<span class="math-tex" data-type="tex">\\(\Gamma = \begin{bmatrix}\frac{1}{2}\Delta t^2 \\ \Delta t\end{bmatrix}\\)</span>

The covariance of the process noise is then

<span class="math-tex" data-type="tex">\\(Q = E[\Gamma w(t) w(t) \Gamma^\mathsf{T}] = \Gamma\sigma^2_v\Gamma^\mathsf{T}\\)</span>.

We can compute that with SymPy as follows

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
var=symbols('sigma^2_v')
v = Matrix([[dt**2/2], [dt]])

Q = v * var * v.T

# factor variance out of the matrix to make it more readable
Q = Q / var
sympy.MatMul(Q, var)
</pre>

The second order system proceeds with the same math.


<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}1 & \Delta t & {\Delta t}^2/2 \\ 0 & 1 & \Delta t\\ 0& 0& 1\end{bmatrix}\\)</span>

Here we will assume that the white noise is a discrete time Wiener process. This gives us

<span class="math-tex" data-type="tex">\\(\Gamma = \begin{bmatrix}\frac{1}{2}\Delta t^2 \\ \Delta t\\ 1\end{bmatrix}\\)</span>

There is no 'truth' to this model, it is just convenient and provides good results. For example, we could assume that the noise is applied to the jerk at the cost of a more complicated equation.

The covariance of the process noise is then

<span class="math-tex" data-type="tex">\\(Q = E[\Gamma w(t) w(t) \Gamma^\mathsf{T}] = \Gamma\sigma^2_v\Gamma^\mathsf{T}\\)</span>.

We can compute that with SymPy as follows

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
var=symbols('sigma^2_v')
v = Matrix([[dt**2/2], [dt], [1]])

Q = v * var * v.T

# factor variance out of the matrix to make it more readable
Q = Q / var
sympy.MatMul(Q, var)
</pre>

We cannot say that this model is more or less correct than the continuous model - both are approximations to what is happening to the actual object. Only experience and experiments can guide you to the appropriate model. In practice you will usually find that either model provides reasonable results, but typically one will perform better than the other.

The advantage of the second model is that we can model the noise in terms of <span class="math-tex" data-type="tex">\\(\sigma^2\\)</span> which we can describe in terms of the motion and the amount of error we expect. The first model requires us to specify the spectral density, which is not very intuitive, but it handles varying time samples much more easily since the noise is integrated across the time period. However, these are not fixed rules - use whichever model (or a model of your own devising) based on testing how the filter performs and/or your knowledge of the behavior of the physical model.

### Using FilterPy to Compute Q

FilterPy offers several routines to compute the <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> matrix. The function `Q_continuous_white_noise()` computes <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> for a given value for <span class="math-tex" data-type="tex">\\(\Delta t\\)</span> and the spectral density.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import filterpy.common as common

common.Q_continuous_white_noise(dim=2, dt=1, spectral_density=1)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
common.Q_continuous_white_noise(dim=3, dt=1, spectral_density=1)
</pre>

The function `Q_discrete_white_noise()` computes <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> assuming a piecewise model for the noise.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
common.Q_discrete_white_noise(2, var=1.)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
common.Q_discrete_white_noise(3, var=1.)
</pre>

### Simplification of Q

Through the early parts of this book I used a much simpler form for <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span>, often only putting a noise term in the lower rightmost element. Is this justified? Well, consider the value of <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> for a small <span class="math-tex" data-type="tex">\\(\Delta t\\)</span>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
common.Q_continuous_white_noise(dim=3, dt=0.05, spectral_density=1)
</pre>

We can see that most of the terms are very small. Recall that the only Kalman filter using this matrix is

<span class="math-tex" data-type="tex">\\( \mathbf{P}=\mathbf{FPF}^\mathsf{T} + \mathbf{Q}\\)</span>

If the values for <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> are small relative to <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span>
than it will be contributing almost nothing to the computation of <span class="math-tex" data-type="tex">\\(\mathbf{P}\\)</span>. Setting <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> to

<span class="math-tex" data-type="tex">\\(Q=\begin{bmatrix}0&0&0\\0&0&0\\0&0&\sigma^2\end{bmatrix}\\)</span>

while not correct, is often a useful approximation. If you do this you will have to perform quite a few studies to guarantee that your filter works in a variety of situations. Given the availability of functions to compute the correct values of <span class="math-tex" data-type="tex">\\(\mathbf{Q}\\)</span> for you I would strongly recommend not using approximations. Perhaps it is justified for quick-and-dirty filters, or on embedded devices where you need to wring out every last bit of performance, and seek to minimize the number of matrix operations required.


** author's note: the following aside probably belongs elsewhere in the book**

> As an aside, most textbooks are more exact with the notation, in Gelb[1] for example, *Pk(+)* is used to denote the uncertainty covariance for the prediction step, and *Pk(-)* for the uncertainty covariance for the update step. Other texts use subscripts with 'k|k-1', superscript <span class="math-tex" data-type="tex">\\(^-\\)</span>, and many other variations. As a programmer I find all of that fairly unreadable; I am used to thinking about variables changing state as a program runs, and do not use a different variable name for each new computation. There is no agreed upon format, so each author makes different choices. I find it challenging to switch quickly between books an papers, and so have adopted this admittedly less precise notation. Mathematicians will write scathing emails to me, but I hope the programmers and students will rejoice.

> If you are a programmer trying to understand a paper's math equations, I strongly recommend just removing all of the superscripts, subscripts, and diacriticals, replacing them with a single letter. If you work with equations like this every day this is superfluous advice, but when I read I am usually trying to understand the flow of computation. To me it is far more understandable to remember that <span class="math-tex" data-type="tex">\\(P\\)</span> in this step represents the updated value of <span class="math-tex" data-type="tex">\\(P\\)</span> computed in the last step, as opposed to trying to remember what <span class="math-tex" data-type="tex">\\(P_{k-1}(+)\\)</span> denotes, and what its relation to <span class="math-tex" data-type="tex">\\(P_k(-)\\)</span> is, if any.

> For example, for the equation of <span class="math-tex" data-type="tex">\\(\mathbf{S}\\)</span> above, Wikipedia uses

> <span class="math-tex" data-type="tex">\\(\textbf{S}_k = \textbf{H}_k \textbf{P}_{k\mid k-1} \textbf{H}_k^\mathsf{T} + \textbf{R}_k
\\)</span>

> Is that more exact? Absolutely. Is it easier or harder to read? You'll need to answer that for yourself.

> For reference, the Appendix **Symbols and Notations** lists the symbology used by the major authors in the field.

## Numeric Integration of Differential Equations

** author's note: this is just notes to a section. If you need to know this in depth,
*Computational Physics in Python * by Dr. Eric Ayars is excellent, and available here.
http://phys.csuchico.edu/ayars/312/Handouts/comp-phys-python.pdf **

So far in this book we have been working with systems that can be expressed with simple linear differential equations such as

<span class="math-tex" data-type="tex">\\(v = \dot{x} = \frac{dx}{dt}\\)</span>

which we can integrate into a closed form solution, in this case <span class="math-tex" data-type="tex">\\(x(t) =vt + x_0\\)</span>. This equation is then put into the system matrix <span class="math-tex" data-type="tex">\\(\mathbf{F}\\)</span>, which allows the Kalman filter equations to predict the system state in the future. For example, our constant velocity filters use

<span class="math-tex" data-type="tex">\\(\mathbf{F} = \begin{bmatrix}
1 & \Delta t \\ 0 & 1\end{bmatrix}\\)</span>.

The Kalman filter predict equation is <span class="math-tex" data-type="tex">\\(\mathbf{x}^- = \mathbf{Fx} + \mathbf{Bu}\\)</span>. Hence the prediction is

<span class="math-tex" data-type="tex">\\(\mathbf{x}^- = \begin{bmatrix}
1 & \Delta t \\ 0 & 1\end{bmatrix}\begin{bmatrix}
x\\ \dot{x}\end{bmatrix}
\\)</span>

which multiplies out to
<span class="math-tex" data-type="tex">\\(\begin{aligned}x^- &= x + v\Delta t \\
\dot{x}^- &= \dot{x}\end{aligned}\\)</span>.

This works for linear ordinary differential equations (ODEs), but does not work (well) for nonlinear equations. For example, consider trying to predict the position of a rapidly turning car. Cars turn by pivoting the front wheels, which cause the car to pivot around the rear axle. Therefore the path will be continuously varying and a linear prediction will necessarily produce an incorrect value. If the change in the system is small enough relative to <span class="math-tex" data-type="tex">\\(\Delta t\\)</span> this can often produce adequate results, but that will rarely be the case with the nonlinear Kalman filters we will be studying in subsequent chapters. Another problem is that even trivial systems produce differential equations for which finding closed form solutions is difficult or impossible.

For these reasons we need to know how to numerically integrate differential equations. This can be a vast topic, and SciPy provides integration routines such as `scipy.integrate.ode`. These routines are robust, but

** material about Euler here**

### Runge Kutta Methods


Runge Kutta integration is the workhorse of numerical integration. As mentioned earlier there are a vast number of methods and literature on the subject. In practice, using the Runge Kutta algorithm that I present here will solve most any problem you will face. It offers a very good balance of speed, precision, and stability, and it is used in vast amounts of scientific software.

Let's just dive in. We start with some differential equation

<span class="math-tex" data-type="tex">\\(
\ddot{y} = \frac{d}{dt}\dot{y}\\)</span>.

We can substitute the derivative of y with a function f, like so

<span class="math-tex" data-type="tex">\\(\ddot{y} = \frac{d}{dt}f(y,t)\\)</span>.

<span class="math-tex" data-type="tex">\\(t(t+\Delta t) = y(t) + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4) + O(\Delta t^4)\\)</span>

<span class="math-tex" data-type="tex">\\(\begin{aligned}
k_1 &= f(y,t)\Delta t \\
k_2 &= f(y+\frac{1}{2}k_1, t+\frac{1}{2}\Delta t)\Delta t \\
k_3 &= f(y+\frac{1}{2}k_2, t+\frac{1}{2}\Delta t)\Delta t \\
k_4 &= f(y+k_3, t+\Delta t)\Delta t
\end{aligned}
\\)</span>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def runge_kutta4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.
    y is the initial value for y
    x is the initial value for x
    dx is the difference in x (e.g. the time step)
    f is a callable function (y, x) that you supply to compute dy/dx for
      the specified values.
    """

    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.
</pre>

Let's use this for a simple example. Let

<span class="math-tex" data-type="tex">\\(\dot{y} = t\sqrt{y(t)}\\)</span>

with the initial values

<span class="math-tex" data-type="tex">\\(\begin{aligned}t_0 &= 0\\y_0 &= y(t_0) = 1\end{aligned}\\)</span>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import math
import numpy as np
t = 0.
y = 1.
dt = .1

ys = []
ts = []

def func(y,t):
    return t*math.sqrt(y)

while t <= 10:
    y = runge_kutta4(y, t, dt, func)
    t += dt

    ys.append(y)
    ts.append(t)

exact = [(t**2+4)**2 / 16. for t in ts]
plt.plot(ts, ys)
plt.plot(ts, exact)
plt.show()

error = np.array(exact) - np.array(ys)
print("max error {}".format(max(error)))
</pre>

## Iterative Least Squares for Sensor Fusion

A broad category of use for the Kalman filter is *sensor fusion*. For example, we might have a position sensor and a velocity sensor, and we want to combine the data from both to find an optimal estimate of state. In this section we will discuss a different case, where we have multiple sensors providing the same type of measurement.

 The Global Positioning System (GPS) is designed so that at least 6 satellites are in view at any time at any point on the globe. The GPS receiver knows the location of the satellites in the sky relative to the Earth. At each epoch (instant in time) the receiver gets a signal from each satellite from which it can derive the *pseudorange* to the satellite. In more detail, the GPS receiver gets a signal identifying the satellite along with the time stamp of when the signal was transmitted. The GPS satellite has an atomic clock on board so this time stamp is extremely accurate. The signal travels at the speed of light, which is constant in a vacuum, so in theory the GPS should be able to produce an extremely accurate distance measurement to the measurement by measuring how long the signal took to reach the receiver. There are several problems with that. First, the signal is not traveling through a vacuum, but through the atmosphere. The atmosphere causes the signal to bend, so it is not traveling in a straight line. This causes the signal to take longer to reach the receiver than theory suggests. Second, the on board clock on the GPS *receiver* is not very accurate, so deriving an exact time duration is nontrivial. Third, in many environments the signal can bounce off of buildings, trees, and other objects, causing either a longer path or *multipaths*, in which case the receive receives both the original signal from space and the reflected signals.

Let's look at this graphically. I will due this in 2D just to make it easier to graph and see, but of course this will generalize to three dimensions. We know the position of each satellite and the range to each (the range is called the *pseudorange*; we will discuss why later). We cannot measure the range exactly, so there is noise associated with the measurement, which I have depicted with the thickness of the lines. Here is an example of four pseudorange readings from four satellites. I positioned them in a configuration which is unlikely for the actual GPS constellation merely to make the intersections easy to visualize. Also, the amount of error shown is not to scale with the distances, again to make it easier to see.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import ukf_internal
with book_format.figsize(10,6):
    ukf_internal.show_four_gps()
</pre>

In 2D two measurements are typically enough to determine a unique solution. There are two intersections of the range circles, but usually the second intersection is not physically realizable (it is in space, or under ground). However, with GPS we also need to solve for time, so we would need a third measurement to get a 2D position.

However, since GPS is a 3D system we need to solve for the 3 dimensions of space, and 1 dimension of time. That is 4 unknowns, so in theory with 4 satellites we have all the information we need. However, we normally have at least 6 satellites in view, and often more than 6. This means the system is *overdetermined*. Finally, because of the noise in the measurements none of pseudoranges intersect exactly.

If you are well versed in linear algebra you know that this an extremely common problem in scientific computing, and that there are various techniques for solving overdetermined systems. Probably the most common approach used by GPS receivers to find the position is the *iterative least squares* algorithm, commonly abbreviated ILS. As you know, if the errors are Gaussian then the least squares algorithm finds the optimal solution. In other words, we want to minimize the square of the residuals for an overdetermined system.

Let's start with some definitions which should be familiar to you. First, we define the innovation as

<span class="math-tex" data-type="tex">\\(\delta \mathbf{z}^-= \mathbf{z} - h(\mathbf{x}^-)\\)</span>

where <span class="math-tex" data-type="tex">\\(\mathbf{z}\\)</span> is the measurement, <span class="math-tex" data-type="tex">\\(h(\bullet)\\)</span> is the measurement function, and <span class="math-tex" data-type="tex">\\(\delta \mathbf{z}^-\\)</span> is the innovation, which we abbreviate as <span class="math-tex" data-type="tex">\\(y\\)</span> in FilterPy. I don't use the <span class="math-tex" data-type="tex">\\(\mathbf{x}^-\\)</span> symbology often, but it is the prediction for the state variable. In other words, this is just the equation <span class="math-tex" data-type="tex">\\(\mathbf{y} = \mathbf{z} - \mathbf{Hx}\\)</span> in the linear Kalman filter's update step.

Next, the *measurement residual* is

<span class="math-tex" data-type="tex">\\(\delta \mathbf{z}^+ = \mathbf{z} - h(\mathbf{x}^+)\\)</span>

I don't use the plus superscript much because I find it quickly makes the equations unreadable, but <span class="math-tex" data-type="tex">\\(\mathbf{x}^+\\)</span> it is just the *a posteriori* state estimate, which is just the predicted or unknown future state. In other words, the predict step of the linear Kalman filter computes this value. Here it is stands for the value of x which the ILS algorithm will compute on each iteration.

These equations give us the following linear algebra equation:

<span class="math-tex" data-type="tex">\\(\delta \mathbf{z}^- = \mathbf{H}\delta \mathbf{x} + \delta \mathbf{z}^+\\)</span>

<span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> is our measurement function, defined as

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \frac{d\mathbf{h}}{d\mathbf{x}} = \frac{d\mathbf{z}}{d\mathbf{x}}\\)</span>

We find the minimum of an equation by taking the derivative and setting it to zero. In this case we want to minimize the square of the residuals, so our equation is

<span class="math-tex" data-type="tex">\\( \frac{\partial}{\partial \mathbf{x}}({\delta \mathbf{z}^+}^\mathsf{T}\delta \mathbf{z}^+) = 0,\\)</span>

where

<span class="math-tex" data-type="tex">\\(\delta \mathbf{z}^+=\delta \mathbf{z}^- - \mathbf{H}\delta \mathbf{x}.\\)</span>

Here I have switched to using the matrix <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> as the measurement function. We want to use linear algebra to peform the ILS, so for each step we will have to compute the matrix <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> which corresponds to <span class="math-tex" data-type="tex">\\(h(\mathbf{x^-})\\)</span> during each iteration.  <span class="math-tex" data-type="tex">\\(h(\bullet)\\)</span> is usually nonlinear for these types of problems so you will have to linearize it at each step (more about this soon).

For various reasons you may want to weigh some measurement more than others. For example, the geometry of the problem might favor orthogonal measurements, or some measurements may be more noisy than others. We can do that with the equation

<span class="math-tex" data-type="tex">\\( \frac{\partial}{\partial \mathbf{x}}({\delta \mathbf{z}^+}^\mathsf{T}\mathbf{W}\delta \mathbf{z}^+) = 0\\)</span>

If we solve the first equation for <span class="math-tex" data-type="tex">\\({\delta \mathbf{x}}\\)</span> (the derivation is shown in the next section) we get

<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}} = {{(\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}}\mathbf{H}^\mathsf{T} \delta \mathbf{z}^-}
\\)</span>

And the second equation yields

<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}} = {{(\mathbf{H}^\mathsf{T}\mathbf{WH})^{-1}}\mathbf{H}^\mathsf{T}\mathbf{W} \delta \mathbf{z}^-}
\\)</span>

Since the equations are overdetermined we cannot solve these equations exactly so we use an iterative approach. An initial guess for the position is made, and this guess is used to compute  for <span class="math-tex" data-type="tex">\\(\delta \mathbf{x}\\)</span> via the equation above. <span class="math-tex" data-type="tex">\\(\delta \mathbf{x}\\)</span> is added to the intial guess, and this new state is fed back into the equation to produce another <span class="math-tex" data-type="tex">\\(\delta \mathbf{x}\\)</span>. We iterate in this manner until the difference in the measurement residuals is suitably small.

### Derivation of ILS Equations (Optional)

I will implement the ILS in code, but first let's derive the equation for <span class="math-tex" data-type="tex">\\(\delta \mathbf{x}\\)</span>. You can skip the derivation if you want, but it is somewhat instructive and not too hard if you know basic linear algebra and partial differential equations.

Substituting <span class="math-tex" data-type="tex">\\(\delta \mathbf{z}^+=\delta \mathbf{z}^- - \mathbf{H}\delta \mathbf{x}\\)</span> into the partial differential equation we get

<span class="math-tex" data-type="tex">\\( \frac{\partial}{\partial \mathbf{x}}(\delta \mathbf{z}^- -\mathbf{H} \delta \mathbf{x})^\mathsf{T}(\delta \mathbf{z}^- - \mathbf{H} \delta \mathbf{x})=0\\)</span>

which expands to

<span class="math-tex" data-type="tex">\\( \frac{\partial}{\partial \mathbf{x}}({\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\mathbf{H}\delta \mathbf{x} -
{\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\delta \mathbf{z}^- -
{\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}\delta \mathbf{x} +
{\delta \mathbf{z}^-}^\mathsf{T}\delta \mathbf{z}^-)=0\\)</span>

We know that

<span class="math-tex" data-type="tex">\\(\frac{\partial \mathbf{A}^\mathsf{T}\mathbf{B}}{\partial \mathbf{B}} = \frac{\partial \mathbf{B}^\mathsf{T}\mathbf{A}}{\partial \mathbf{B}} = \mathbf{A}^\mathsf{T}\\)</span>

Therefore the third term can be computed as

<span class="math-tex" data-type="tex">\\(\frac{\partial}{\partial \mathbf{x}}{\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}\delta \mathbf{x} = {\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}\\)</span>

and the second term as

<span class="math-tex" data-type="tex">\\(\frac{\partial}{\partial \mathbf{x}}{\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\delta \mathbf{z}^-={\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}\\)</span>

We also know that
<span class="math-tex" data-type="tex">\\(\frac{\partial \mathbf{B}^\mathsf{T}\mathbf{AB}}{\partial \mathbf{B}} = \mathbf{B}^\mathsf{T}(\mathbf{A} + \mathbf{A}^\mathsf{T})\\)</span>

Therefore the first term becomes

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\frac{\partial}{\partial \mathbf{x}}{\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\mathbf{H}\delta \mathbf{x} &= {\delta \mathbf{x}}^\mathsf{T}(\mathbf{H}^\mathsf{T}\mathbf{H} + {\mathbf{H}^\mathsf{T}\mathbf{H}}^\mathsf{T})\\
&= {\delta \mathbf{x}}^\mathsf{T}(\mathbf{H}^\mathsf{T}\mathbf{H} + \mathbf{H}^\mathsf{T}\mathbf{H}) \\
&= 2{\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\mathbf{H}
\end{aligned}\\)</span>

Finally, the fourth term is

<span class="math-tex" data-type="tex">\\( \frac{\partial}{\partial \mathbf{x}}
{\delta \mathbf{z}^-}^\mathsf{T}\delta \mathbf{z}^-=0\\)</span>

Replacing the terms in the expanded partial differential equation gives us

<span class="math-tex" data-type="tex">\\(
 2{\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\mathbf{H} -
 {\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H} - {\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}
 =0
\\)</span>

<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\mathbf{H} -
 {\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H} = 0\\)</span>

 <span class="math-tex" data-type="tex">\\({\delta \mathbf{x}}^\mathsf{T}\mathbf{H}^\mathsf{T}\mathbf{H} =
 {\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}\\)</span>

Multiplying each side by <span class="math-tex" data-type="tex">\\((\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}\\)</span> yields

<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}}^\mathsf{T} =
{\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}(\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}\\)</span>

Taking the transpose of each side gives

<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}} = ({{\delta \mathbf{z}^-}^\mathsf{T}\mathbf{H}(\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}})^\mathsf{T} \\
={{(\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}}^T\mathbf{H}^\mathsf{T} \delta \mathbf{z}^-} \\
={{(\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}}\mathbf{H}^\mathsf{T} \delta \mathbf{z}^-}
\\)</span>

For various reasons you may want to weigh some measurement more than others. We can do that with the equation

<span class="math-tex" data-type="tex">\\( \frac{\partial}{\partial \mathbf{x}}({\delta \mathbf{z}}^\mathsf{T}\mathbf{W}\delta \mathbf{z}) = 0\\)</span>

Replicating the math above with the added <span class="math-tex" data-type="tex">\\(\mathbf{W}\\)</span> term results in

<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}} = {{(\mathbf{H}^\mathsf{T}\mathbf{WH})^{-1}}\mathbf{H}^\mathsf{T}\mathbf{W} \delta \mathbf{z}^-}
\\)</span>

### Implementing Iterative Least Squares

Our goal is to implement an iterative solution to
<span class="math-tex" data-type="tex">\\({\delta \mathbf{x}} = {{(\mathbf{H}^\mathsf{T}\mathbf{H})^{-1}}\mathbf{H}^\mathsf{T} \delta \mathbf{z}^-}
\\)</span>

First, we have to compute <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span>, where <span class="math-tex" data-type="tex">\\(\mathbf{H} =  d\mathbf{z}/d\mathbf{x}\\)</span>. Just to keep the example small so the results are easier to interpret we will do this in 2D. Therefore for <span class="math-tex" data-type="tex">\\(n\\)</span> satellites <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> expands to

<span class="math-tex" data-type="tex">\\(\mathbf{H} = \begin{bmatrix}
\frac{\partial p_1}{\partial x_1} & \frac{\partial p_1}{\partial y_1} \\
\frac{\partial p_2}{\partial x_2} & \frac{\partial p_2}{\partial y_2} \\
\vdots & \vdots \\
\frac{\partial p_n}{\partial x_n} & \frac{\partial p_n}{\partial y_n}
\end{bmatrix}\\)</span>

We will linearize <span class="math-tex" data-type="tex">\\(\mathbf{H}\\)</span> by computing the partial for <span class="math-tex" data-type="tex">\\(x\\)</span> as

<span class="math-tex" data-type="tex">\\( \frac{estimated\_x\_position - satellite\_x\_position}{estimated\_range\_to\_satellite}\\)</span>

The equation for <span class="math-tex" data-type="tex">\\(y\\)</span> just substitutes <span class="math-tex" data-type="tex">\\(y\\)</span> for <span class="math-tex" data-type="tex">\\(x\\)</span>.

Then the algorithm is as follows.

    def ILS:
        guess position
        while not converged:
            compute range to satellites for current estimated position
            compute H linearized at estimated position
            compute new estimate delta from (H^T H)'H^T dz
            new estimate = current estimate + estimate delta
            check for convergence

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy as np
from numpy.linalg import norm, inv
from numpy.random import randn
from numpy import dot


np.random.seed(1234)
user_pos = np.array([800, 200])


sat_pos = np.asarray(
    [[0, 1000],
     [0, -1000],
     [500, 500]], dtype=float)

def satellite_range(pos, sat_pos):
    """ Compute distance between position 'pos' and the list of positions
    in sat_pos"""

    N = len(sat_pos)
    rng = np.zeros(N)

    diff = np.asarray(pos) - sat_pos

    for i in range(N):
        rng[i] = norm(diff[i])

    return norm(diff, axis=1)


def hx_ils(pos, sat_pos, range_est):
    """ compute measurement function where
    pos : array_like
        2D current estimated position. e.g. (23, 45)

    sat_pos : array_like of 2D positions
        position of each satellite e.g. [(0,100), (100,0)]

    range_est : array_like of floats
        range to each satellite
    """

    N = len(sat_pos)
    H = np.zeros((N, 2))
    for j in range(N):
        H[j,0] = (pos[0] - sat_pos[j,0]) / range_est[j]
        H[j,1] = (pos[1] - sat_pos[j,1]) / range_est[j]
    return H


def lop_ils(zs, sat_pos, pos_est, hx, eps=1.e-6):
    """ iteratively estimates the solution to a set of measurement, given
    known transmitter locations"""
    pos = np.array(pos_est)

    with book_format.numpy_precision(precision=4):
        converged = False
        for i in range(20):
            r_est = satellite_range(pos, sat_pos)
            print('iteration:', i)

            H=hx(pos, sat_pos, r_est)
            Hinv = inv(dot(H.T, H)).dot(H.T)

            #update position estimate
            y = zs - r_est
            print('innovation', y)

            Hy = np.dot(Hinv, y)
            pos = pos + Hy
            print('pos       {}\n\n'.format(pos))

            if max(abs(Hy)) < eps:
                converged = True
                break

    return pos, converged

# compute measurement of where you are with respect to each sensor
rz= satellite_range(user_pos, sat_pos)

pos, converted = lop_ils(rz, sat_pos, (900,90), hx=hx_ils)
print('Iterated solution: ', pos)
</pre>

So let's think about this. The first iteration is essentially performing the computation that the linear Kalman filter computes during the update step:

<span class="math-tex" data-type="tex">\\(\mathbf{y} = \mathbf{z} - \mathbf{Hx}\\
\mathbf{x} = \mathbf{x} + \mathbf{Ky}\\)</span>

where the Kalman gain equals one. You can see that despite the very inaccurate initial guess (900, 90) the computed value for <span class="math-tex" data-type="tex">\\(\mathbf{x}\\)</span>, (805.4, 205.3), was very close to the actual value of (800, 200). However, it was not perfect. But after three iterations the ILS algorithm was able to find the exact answer. So hopefully it is clear why we use ILS instead of doing the sensor fusion with the Kalman filter - it gives a better result. Of course, we started with a very inaccurate guess; what if the guess was better?

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
pos, converted = lop_ils(rz, sat_pos, (801, 201), hx=hx_ils)
print('Iterated solution: ', pos)
</pre>

The first iteration produced a better estimate, but it still could be improved upon by iterating.

I injected no noise in the measurement to test and display the theoretical performance of the filter. Now let's see how it performs when we inject noise.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# add some noise
nrz = []
for z in rz:
    nrz.append(z + randn())
pos, converted = lop_ils(nrz, sat_pos, (601,198.3), hx=hx_ils)
print('Iterated solution: ', pos)
</pre>

Here we can see that the noise means that we no longer find the exact solution but we are still able to quickly converge onto a more accurate solution than the first iteration provides.

This is far from a complete coverage of the iterated least squares algorithm, let alone methods used in GNSS to compute positions from GPS pseudoranges. You will find a number of approaches in the literature, including QR decomposition, SVD, and other techniques to solve the overdetermined system. For a nontrivial task you will have to survey the literature and perhaps design your algorithm depending on your specific sensor configuration, the amounts of noise, your accuracy requirements, and the amount of computation you can afford to do.

## References

 * [1] *Matrix Exponential* http://en.wikipedia.org/wiki/Matrix_exponential

 * [2] *LTI System Theory* http://en.wikipedia.org/wiki/LTI_system_theory

 * [3] C.F. van Loan, "Computing Integrals Involving the Matrix Exponential," IEEE Transactions Automatic Control, June 1978.

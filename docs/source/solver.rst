Solving the walk
================

To determine the solution to the set of ODEs defined in equation :eq:`ode`, we employ a straightforward Euler scheme.
Time is discretized into steps of :math:`\Delta t`, and the evolution is expressed as:

.. math::

    x_{i, n + 1} = x_{i, n} + \Delta t \, \dot{x}_{i,n}

    \dot{x}_{i, n + 1} = \dot{x}_{i, n} + \ddot{x}_{i, n} \, \Delta t \, .


In these equations, :math:`x_{i, n}` represents the position of walker :math:`i` at time :math:`t = n \Delta t`, and a similar notation is employed for velocity and acceleration.
The acceleration, of course, is computed using equation :eq:`ode`.

In our library, these calculations are carried out by the Solver class, which simplifies the process significantly.
For instance, if we aim to find the solution at a specific time, say :math:`t = 1` second, the code might look like this:

.. code::

    import mockWalkers as mckw

    # Define the problem ...
    # walkers = ...
    # delta_t = ...
    # ...

    s = mckw.Solver(
        walkers,
        delta_t,
        vd_calcs,
        obstacles,
        geometry,
    )

    while s.current_time < 1:
        s.iterate()
    
    # Do something with the solution ...

The various elements used for describing the problem will be explained in detail later.
What's crucial to note here is the :code:`Solver.iterate` method.
Each time this method is called, it performs one iteration of the Euler scheme, advancing the state from :math:`t = n \Delta t` to :math:`t = (n + 1) \Delta t`.
This approach simplifies the numerical solution of the ODEs, allowing you to obtain the desired solution at any desired time point.
Walking model
=============

Let :math:`x_i \left( t \right) = \left( X_i \left( t \right), Y_i \left( t \right) \right)` be the position of a walker :math:`i` at time  :math:`t > 0` in a crowd of :math:`N` 
individuals  (i.e. :math:`i = 1, ... , N`).

Our modeling approach relies on a set of Newton-like Ordinary Differential Equations (ODEs) to describe the dynamics:

.. math::
    :label: ode
    
    \ddot{x}_i = F ( x_i, \dot{x}_i ) + \sum_{i \neq j, i = 1}^{N} K (x_i, x_j ) + E(x_i) \, .

Positions and velocities are intended in the 2D plane.
Here are the key components:

* :math:`F` regulates propulsion as :math:`F = \frac{v_d ( x_i ) - \dot{x}_i}{\tau}`, where :math:`v_d ( x_i )` is a “desired velocity field” and :math:`\tau` is a relaxation time.

* :math:`K` is a pairwise interaction kernel, of form :math:`K ( x_i,x_j ) = A \exp \left( \frac{- \left| x_i - x_j \right|^2}{R^2} \right) \frac{x_i - x_j}{\left| x_i - x_j \right|} \theta ( x_i - x_j, v_d)`. It embodies a repulsion force from :math:`x_j` to :math:`x_i`, which comes into play only when :math:`x_j` is within the view cone of :math:`x_i`. The parameter :math:`R` represents the typical interaction radius, and the view cone is determined by :math:`\theta ( x_i - x_j, v_d)`, which is zero if the angle between :math:`v_d` and :math:`x_i - x_j` exceeds a certain threshold (e.g., 80 degrees) and 1 otherwise.

* :math:`E` accounts for the repulsion caused by obstacles. It is represented as :math:`E = B \sum_{k} \exp \left(- \frac{d_{i,k}}{R_k^\prime} \right) \vec{n_k}`, where :math:`d_{i,k}` is the distance between the walker :math:`i` and the obstacle :math:`k`, :math:`\vec{n}` is the normal vector pointing towards the obstacle and :math:`R^\prime` is a distance scale representing the "impermeability" of the obstacle.

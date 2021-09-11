# SADI

Sketch-and-project ADI (SADI) methods for solving the Peaceman-Rachford problem and Sylvester matrix equations.

## ADI model problems

SADI implements sketch-and-project version of the Peaceman-Rachford (PR) method, that we call *SPR*, which solves in $u$ problems of the form:
.. math::
    (H + V) u = s

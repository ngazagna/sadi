SADI
=====


Sketch-and-project variants of ADI (SADI) methods for solving the Peaceman-Rachford problem and Sylvester matrix equations.

SADI implements sketch-and-project version of the Peaceman-Rachford (PR) method, which solves problems of the form:
.. math::

    (H + V) u = s

where .. math::`H` and .. math::`V` are accessible alternatively, and of the Alternating-Direction Implicit (ADI) method which is well designed which solves Sylvester matrix equations:
.. math::

    AX - XB = F

At each iteration, instead of solving an entire (shifted) system we project the previous iterate on a sketched/subsampled version of the linear system to solve.


Dependencies
============

All codes are in Python 3 and experiments are runnable from jupyter notebooks.

All dependencies are in **TODO**: ``./environment.yml``

Cite
====

* **TODO**: add the arxiv link here

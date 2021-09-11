# SADI

Sketch-and-project ADI (SADI) methods for solving the Peaceman-Rachford problem and Sylvester matrix equations.

## ADI model problems

SADI implements sketch-and-project version of the Peaceman-Rachford (PR) method, that we call *SPR*, which solves in $u$ problems of the form:
```math
(H + V) u = s
```
where $H$ and $V$ are accessible alternatively, and of the Alternating-Direction Implicit (ADI) method which is well designed which solves in $X$ Sylvester matrix equations:
```math
AX - XB = F
```
At each iteration, instead of solving an entire (shifted) system we project the previous iterate on a sketched/subsampled version of the linear system to solve.



# Implemented method

## 1) Peaceman-Rachford (PR) method
We solve
$$(H+V)u=s$$
using
- vanilla **PR** method implemented in function `adi_pr`
- **BSPR**, **B**lock **S**ketch-and-project **PR** algorithm with row block sketching, implemented in function `sap_adi_pr`

in the `adi_peaceman_rachford.py` file. As shift parameters, this two solvers can be passed
- nothing: in this case they are set to zero, *ie* $\, p_j = q_j = 0$
- single values: in this case they are constant, *ie* $\, p_j = p\,$ & $\, q_j = q$
- arrays: *ie* $\, p_j = p[j] \,$ & $\, q_j = q[j]$

$\forall j=0, ..., n_{iter}-1$

Two types of efficient shift parameters are available in function: `pr_shift_parameters` and `wachspress_shift_parameters` in the `adi_pr.py` file.

## 2) Alternating-Direction Implicit (ADI) method for Sylvester matrix equations
We solve
$$AX - XB = C$$
using
- vanilla **ADI** implemented in function `adi_syl`
- **BSADI**, **B**lock Sketch-and-project ADI with row/column block sketching, implemented in function `sap_adi_pr`

in the `adi_sylvester.py` file.

Like for PR problem, user can pass shifts to this solvers.


# Requirements
Python 3 (with basic packages `numpy`, `scipy`, `matplotlib`)

All dependencies are in **TODO**: ``./environment.yml``


# Cite

* **TODO**: add the arxiv link here

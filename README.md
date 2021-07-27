# SDFL
SOFT-DISKS FLUID LEARNING

AUTHOR: LUCA ZAMMATARO, 2020

DESCRIPTION: 
This work is based on the Equivalence between Molecular Dynamics and Neural Network. It provides
learning proofs in a Lennard-Jones (LJ) fluid, presented as a network of particles having non-bonded interactions. I
describe the fluid's learning as the property of an order that emerges as an adaptation in establishing equilibrium with
energy and thermal conservation. The experimental section demonstrates the fluid can be trained with logic-gates
patterns. The work goes beyond Molecular Computing's application, explaining how this model uses its intrinsic
minimizing properties in learning and predicting outputs. Finally, it gives hints for a theory on real chemistry's
computational universality.

LICENSE:
Copyright (c) 2020 Luca Zammataro

Usage: python LJSDF.py <training/testing file>

I.e.:  python LJSDF.py AND.txt

Note: 
To "training" the algorithm with a logic-gate, use as argument one of the text files provided (AND.txt, NAND.txt OR.txt, NOR.txt XOR.txt, and XNOR.txt)

To "testing" a logic-gate, copy the 3 weights files (weights.r, weights.rv, and weights.ra) from the logic-gate directory you want to test, into the same position where you're launching the python script. Use as argument one of the text files into the LOGIC_PATTERNS directories.

The code is described in:

Molecular Learning of a Soft-Disks Fluid
Luca Zammataro
bioRxiv 2021.07.24.453642; doi: https://doi.org/10.1101/2021.07.24.453642


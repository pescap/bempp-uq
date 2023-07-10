# Bempp-UQ
A Python/Bempp-based plug-in for uncertainty quantification by random domains. 
Refer to: Helmholtz scattering by random domains: first-order sparse boundary element approximation

```
@article{escapil2020helmholtz,
  title={Helmholtz scattering by random domains: first-order sparse boundary element approximation},
  author={Escapil-Inchausp{\'e}, Paul and Jerez-Hanckes, Carlos},
  journal={SIAM Journal on Scientific Computing},
  volume={42},
  number={5},
  pages={A2561--A2592},
  year={2020},
  publisher={SIAM}
}
```

Includes:
- (FOA) Tutorials to evaluate the shape derivative for both sound-soft and sound-hard Helmholtz scattering problems
- (TP Main) Tutorial to solve the Transmission Problem (TP) and analyze the convergence of GMRES
- (Combination Technique) Implementation of the CT for the TP by the unit sphere (work in progress to adapt the structure to the docker version of bempp)
- (Tensor Operator Solver) Solve preconditioned tensor operator equations with GMRES
- projection.py Routines to evaluate the transfer operators between grids for P1 elements
- login.py Minor modifications of the original Bempp code

A Bempp Docker image is easily accessible at: https://bempp.com/download/docker/

---
![alt text](https://i.ibb.co/YQ9Bqpt/transformation.png)*<p align="center">Example of transformed boundaries respect to t meshed with 3,249 vertices</p>*.

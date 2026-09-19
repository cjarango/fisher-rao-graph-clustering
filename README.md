# Fisher-Rao Graph Clustering on Statistical Manifolds

## Context

Data analysis on information manifolds is a problem in geometric statistics and manifold learning. In particular, Gamma distributions form a two-dimensional Riemannian manifold equipped with the Fisher-Rao metric, whose negative sectional curvature makes it a Hadamard manifold, guaranteeing the existence of a unique minimizing geodesic between any two points.

Computing these geodesics requires solving a boundary value problem (BVP) associated with the Euler-Lagrange equations, which becomes prohibitive when the full distance matrix between $N$ observations is needed, since its complexity grows as $\mathcal{O}(N^2)$.

This thesis project proposes and evaluates an alternative: approximating geodesic distances through a neighborhood graph built on the observed distributions, and comparing its performance in clustering tasks (PAM) against the exact ODE-based computation.

## Objectives

- **Exact reference computation:** Solve the geodesic equations on the Gamma manifold via a BVP solver, obtaining the exact distance matrix.
- **Graph-based approximation:** Build a k-NN graph over the distributions, with neighbor selection via symmetric KL divergence and edge weights computed using the local Fisher-Rao metric.
- **Comparative clustering:** Apply PAM and HDBSCAN to both dissimilarity matrices and compare clustering quality using the Adjusted Rand Index.
- **Product manifolds:** Extend the framework to product manifolds, combining component-wise geodesic distances through the product metric.
- **Complexity analysis:** Estimate the time complexity of each approach and demonstrate the graph's advantage under dense sampling regimes.

*Future work:* Extension of the framework to non-parametric information manifolds, where the Fisher-Rao geometry is defined over infinite-dimensional spaces of distributions rather than finite-dimensional parametric families.

## Project structure

```bash
.
├── .github/
├── data/
├── diagrams/
│   └── architecture_uml_detailed.svg
├── include/
│   └── frgc/
│       └── version.hpp
├── src/
│   ├── app/
│   │   └── main.cpp
│   └── core/
│       └── version.cpp
├── tests/
│   └── test_openmp.cpp
├── .dockerignore
├── .gitignore
├── CMakeLists.txt
├── Dockerfile
├── environment.yml
├── LICENSE
└── README.md
```

## Requirements

- Docker Desktop (or Docker Engine on Linux)
- Git

*Note:* There is no need to install compilers, CMake, or OpenMP on the host system: the entire toolchain lives inside the Docker image.

## Installation and setup

**1. Clone the repository:**

```bash
git clone https://github.com/your-user/fisher-rao-graph-clustering.git
cd fisher-rao-graph-clustering
```

**2. Build the Docker image:**

```bash
docker build -t frgc-dev .
```

This step pulls Debian Bookworm Slim, installs micromamba, and creates the `frgc` environment with GCC 15, CMake, Ninja, and OpenMP.

**3. Start the container with the project mounted:**

```bash
docker run --rm -it -v ${PWD}:/app frgc-dev
```

*Note:* On Linux/macOS, replace `${PWD}` with `$(pwd)`.

## Usage

**1. Build the project** inside the container:

```bash
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
```

**2. Run the main binary:**

```bash
./bin/frgc_app
```

**3. Run the test suite:**

```bash
ctest --output-on-failure
```

**4. Measure OpenMP speedup** with different thread counts:

```bash
for t in 1 2 4 8; do
  OMP_PROC_BIND=close OMP_PLACES=cores ./bin/test_openmp $t
done
```

## Technical specifications

The project is built and executed inside a Docker container, ensuring a reproducible environment independent of the host system. The following table summarizes the main components of the toolchain.

<div align="center">

<table style="border-collapse: collapse; border: 1px solid #ffffff; margin: 0 auto;">
  <thead>
    <tr style="border-bottom: 1px solid #ffffff;">
      <th style="border: 1px solid #ffffff; padding: 8px;">Component</th>
      <th style="border: 1px solid #ffffff; padding: 8px;">Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ffffff; padding: 8px;">Language standard</td>
      <td style="border: 1px solid #ffffff; padding: 8px;">C++17</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ffffff; padding: 8px;">Compiler</td>
      <td style="border: 1px solid #ffffff; padding: 8px;">GCC 15.3.0 (conda-forge)</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ffffff; padding: 8px;">Build system</td>
      <td style="border: 1px solid #ffffff; padding: 8px;">CMake &gt;= 3.16 + Ninja</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ffffff; padding: 8px;">Parallelization</td>
      <td style="border: 1px solid #ffffff; padding: 8px;">OpenMP 4.5</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ffffff; padding: 8px;">Environment</td>
      <td style="border: 1px solid #ffffff; padding: 8px;">Docker + micromamba on Debian Bookworm</td>
    </tr>
  </tbody>
</table>

</div>

## License

Distributed under the MIT License. This license permits use, modification, and distribution of the code with few restrictions, provided the copyright notice and permission statement are included. For more details, see the `LICENSE` file.
# Lascon X Project

The main idea is to repdouce the paper of Tim Vogels from 2011 - Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways and Memory Networks ([link](https://www.science.org/doi/10.1126/science.1211095)). 

## Project goals

1. **Reproduce key figures** from Vogels et al. (2011) using NEST.
2. **Deliver a clear NEST example** of inhibitory synaptic plasticity (Vogels rule) that others can reuse.
3. **Document the workflow** end-to-end (network setup → stimulation → learning → measurements → plotting).

## Reasoing

- Learn NEST deeply through a real, canonical replication.
- Understand a widely used inhibitory plasticity mechanism not covered in the school curriculum.
- Contribute a practical, community-friendly example implementation.

## Background and references

- All the supplementary material is availe in the folder [docs](https://github.com/tuliofalmeida/lascon_vogels/tree/main/docs) and also what is already implemented in [ModelDB](https://modeldb.science/143751)
- I really recommend this e-book about coding for nerusocientist, it's usually useful to understand GitHub, environments and good code practices: [The Good Research Code Handbook](https://goodresearch.dev/index.html)
- Our project in [OverLeaf](https://www.overleaf.com/project/696436afd542f478c53c523d) - **Need to be updated**
- NEST [Documentation](https://nest-simulator.readthedocs.io/en/stable/index.html)

## What we replicate

Everything structural in the paper is standard NEST:

- Integrate-and-fire neurons
- Conductance-based synapses
- Poisson external input
- Large recurrent networks (E/I)
- External stimulation (including inhomogeneous Poisson drive)
- Assembly formation via weight scaling
- Spike recording / population rates / balance metrics

Core learning mechanism:
- [Vogels inhibitory plasticity rule](https://nest-simulator.readthedocs.io/en/stable/models/vogels_sprekeler_synapse.html) already exists in NEST, let's wrap it into a reproducible example and figure pipeline

## Expected outcomes

By the end of the project, we expect:

- Reproduction of **at least two main figures** from the paper (see checklist below)
- A **solid, minimal NEST example** that implements the Vogels inhibitory synaptic plasticity rule
- Scripts that generate publication-style plots from saved simulation outputs
- A documented pipeline and parameters aligned with the original study

## Repository structure

Suggested structure for our repository

```
├─ README.md //
├─ environment.yml # As is not that easy to install it, let's let it on hold \\
├─ configs/
│ ├─ default.yaml # baseline parameters
│ ├─ figure1.yaml # parameters to reproduce Figure 1 (example)
│ └─ figure2.yaml # parameters to reproduce Figure 2 (example)
├─ src/
│ ├─ model.py # network construction (neurons, synapses, connect)
│ ├─ stimulation.py # Poisson / inhomogeneous Poisson protocols
│ ├─ plasticity.py # Vogels rule setup helper
│ ├─ measures.py # e.g, E/I balance metrics, rates, CV, correlations
│ ├─ io.py # saving/loading results
│ └─ utils.py
├─ scripts/
│ ├─ run_simulation.py # single-run entrypoint
│ └─ export_figures.py # recreate plots from saved outputs
├─ notebooks/
│ └─ exploratory.ipynb # to test/develop
├─ results/
│ ├─ raw/ # simulation outputs 
│ └─ figures/ # generated figures
└─ tests/
└─ test_smoke.py # Down-scaled parameters to run fast```

# PyBasin Roadmap

## Features

### Classifier Enhancements

- [ ] Allow classifiers to define ODE parameters per initial condition
  - Currently `KNNClassifier` accepts a single `ode_params` for all templates
  - bSTAB allows defining different parameters for each template initial condition
  - This would enable more flexible template matching across parameter spaces
- [ ] When varying parameters we should not vary initial conditions
- [ ] Optimize parameters variation (not hyper-parameters)
- [ ] Improve plotter API. Calling plt.show does not feels right

```python
plotter.plot_templates_trajectories(
  plotted_var=0,
  y_limits=(-1.4, 1.4),
  x_limits=(0, 50),
)
plt.show()  # type: ignore[misc]
```

- [ ] Look into https://github.com/lmcinnes/umap for feature space visualization
- [ ] Rename as_parameter_manager to bs_study_parameter_manager and as_bse to study_bse
- [ ] Fix Installation guideline, find out how to deploy to pip
- [ ] Using JAX SaveAt and setting diffeqsolve.t0 = 0 we can make JAX return the transient time and save a lot of memory. Need to check if that behaviour applies to other solvers. This could help a lot for parameter sweeps with batch integration, saving 50 points intead of 1000 virtually saves 20x space

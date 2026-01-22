# PyBasin Roadmap

## Features

### Classifier Enhancements

- [ ] Allow classifiers to define ODE parameters per initial condition
  - Currently `KNNClassifier` accepts a single `ode_params` for all templates
  - bSTAB allows defining different parameters for each template initial condition
  - This would enable more flexible template matching across parameter spaces

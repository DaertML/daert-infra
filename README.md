# daert-infra
Repo to deploy ML infrastructure with ease at scale

# Introduction
The main idea of this repository is to provide solutions that can perform the common operations that we find in different fields of ML with ease. The ideas of what to deploy and its metaphors are from march 2023, as we are in 2026, you can see things havent really changed that much in the field. The main motivation was to reproduce the AzureML ecosystem at home, and do so with ease, without the burden of having to manually deploy or manage instances. At that time, the main reason why this project was not done was a bit of laziness and not being sure if the metaphors would be radically different as time passed. Nowadays, here we are doing things in the exact same manner.

# Components
The original designs will be saved under the "mlops" folder, and will contain in different folders:
- AutoML training: train multiple ML models with multiple parameter configurations done automatically, given a CSV file with data.
- Pipeline management: Manage data pipelines with ease.
- Notebooks: experiment with ideas in an environment that allows for an easy deployment of notebooks.
- Compute Instance: raw container where you can modify from a set of given blueprints, the needed dependencies, code...

As MLFlow is quite rich nowadays and has support for LLM outputs, Im planning to get some ideas going to do things with LLMs.

# Positional Encoding meets Persistent Homology on Graphs

 [Yogesh Verma](https://yoverma.github.io/yoerma.github.io/) | [Amauri H. Souza](https://www.amauriholanda.org)  |  [Vikas Garg](https://www.mit.edu/~vgarg/)

The repository is developed on the intersection of [RePHINE](https://github.com/Aalto-QuML/RePHINE), and [SPE](https://github.com/Graph-COM/SPE). Please refer to their repos for specific requirements.

** Code will be updated here **


To run the method on ZINC or Alchemy, do the following:

```
cd SPE/zinc/ or SPE/alchemy
python -u runner.py --config_dirpath ../configs/alchemy (or zinc) --config_name SPE_gine_gin_mlp_pe37.yaml --seed 0
```

One can change the type of PE method in the config files.

# Certifiable Robustness of GCN under Structure Perturbations

Implementation of the paper:   
**[Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations](https://dl.acm.org/doi/abs/10.1145/3394486.3403217)**

by Daniel Zügner and Stephan Günnemann.   
Published at KDD'20, August 2020 (virtual event)

Copyright (C) 2020   
Daniel Zügner   
Technical University of Munich    

## Additional resources
[[Paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403217) | [Slides (KDD 2020)](https://www.in.tum.de/fileadmin/w00bws/daml/robust-gcn/kdd_2020_presentation.pdf)]

## Run the code
 
The fastest way to try our code is to use the Jupyter notebook `demo.ipynb`.  
If you would like to reproduce our numbers shown in the paper, use `reproduce.ipynb`. Note that in order to reproduce the results exactly you may need to install the solver [CPLEX](https://www.ibm.com/analytics/cplex-optimizer), which is proprietary. The open-source linear program solver `ECOS` is slower and fails on some of the instances and therefore results in slightly lower numbers than reported in the paper.

## Installation
`python setup.py install`

If you just want to add a symbolic link to your package directory run   
`python setup.py develop`
 
## Contact
Please contact zuegnerd@in.tum.de in case you have any questions.


## References
### Datasets
In the `datasets` folder we provide the following datasets originally published by   
#### Cora
McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie.  
*Automating the construction of internet portals with machine learning.*   
Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of   
attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

#### Citeseer
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.
#### PubMed
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.

### Graph Convolutional Networks
Our implementation of the GCN algorithm is based on the authors' implementation,
available on GitHub [here](https://github.com/tkipf/gcn).

The paper was published as  

Thomas N Kipf and Max Welling. 2017.  
*Semi-supervised classification with graph
convolutional networks.* ICLR (2017).

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{10.1145/3394486.3403217, 
author = {Z\"{u}gner, Daniel and G\"{u}nnemann, Stephan},
title = {Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations},
year = {2020},
isbn = {9781450379984},
doi = {10.1145/3394486.3403217}, 
booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
pages = {1656–1665},
location = {Virtual Event, CA, USA},
series = {KDD '20} }
}
```
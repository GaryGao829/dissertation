1. 
Use command 'python3 cifar10_vgg.py' to train a VGG network from scratch.
It dumps weight in every iteration.

2. 
In file: clustering_trails.ipynb
I tried some clustering methods like: kmeans(with/without PCA), k-shape(one time series clustering method), spectral clustering.
None of above works good. I judge it by clustering them twice, the results are not self-consistent.
Finally I use KDE(kernel density estimation) to locate where the parameter starts/end to converge.
Report here: https://docs.google.com/document/d/1tLMZS8xAh1MKQazhQdPnrkPrTf1mw14f_G8MkUSt12k/edit 

3.
In file: Kernel Density Estimations.ipynb
I used KDE to get convergency start_point for every parameter in ceratin layer. I generated two set of parameters from 'cifar10_vgg.py'. 
It loads the same dataset in the same order. 
The model structure maintains unchanged. 
The only difference is the paramters initialization value. 
But the pattern doesn't show up observed from the KDE method, according to the green dot pictures in ' Kernel Density Estimations.ipynb'

4.
In file 'Final parameters states comparison in 2 trails.ipynb' I compared final parameters in two trails. 
(Here 'final' means after all iterations)
I calculated MSE(mean square error) of all parameters. The value is shown in picture in that file. 
It's much lower than 2 set of random IID numbers(Standard Normal Distribution), whose MSE is around: 2.0
MSE of layer 1,4,5,7: [0.16, 0.12, 0.01, 0.11]

Analysis && Summary ::: 
In trail1 and trail2, I used the same data set, the same model, the same training process(same training-set feeding sequence). 
The only difference is the initial value of model in 2 trails. 
From 4, I find that all parameters have similar convergency situation. Their final parameters' values are similar. 
However, the convergency pattern is still not clear discovered. 

5.
in file 'train cifar10_vgg with same init.ipynb'
I use the same initial value to train the model.
Although I used the same init value to train the model, the parameters are not 100% same in 2 trails.(slightly difference)

6. 
However, pictures based on 5 is not shaped like y=x. 
I doubt the 'pre-processing' function may not correct 

---- todo ----
1. modify pre-processing function;
2. modify plotting function: size(color) of dot should differ based on the number of smae points.


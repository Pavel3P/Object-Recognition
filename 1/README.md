# 1st lab 
### Implement EM for Bernoulli mixtures models


<img src="https://render.githubusercontent.com/render/math?math=K"> - Number of clusters<br/>

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{x_n} = (\mathbf{x_{1},...,x_{D}}), n=\overline{1,N}"> - Sample<br/>

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu_m} = (\mathbf{\mu_{m,1},...,\mu_{m,D}}), m=\overline{1,K}"> - EM algorithm parameters<br/>

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z_n} = (\mathbf{z_{1},...,z_{K}}), n=\overline{1,N}"> - Predictions for nth sample<br/>

#### Formula for expectation step:

<center>
    <img src="https://render.githubusercontent.com/render/math?math=z_{n, k} \leftarrow \frac{\pi_k \prod_{i = 1}^D \mu_{k, i}^{x_{n, i}} (1 - \mu_{k, i})^{1 - x_{n, i}} }{\sum_{m = 1}^K \pi_m \prod_{i = 1}^D \mu_{m, i}^{x_{n, i}} (1 - \mu_{m, i})^{1 - x_{n, i}}} (1)">
</center>

#### Formula for maximization step:

<center>
    <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu_m} \leftarrow \mathbf{\bar{x}_m} (2)"><br/>
    <img src="https://render.githubusercontent.com/render/math?math=\pi_m \leftarrow \frac{N_m}{N} (3)"><br/>
    Where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\bar{x}_m} = \frac{1}{N_m} \sum_{n = 1}^N z_{n, m} \mathbf{x_n} (4)">, and
    <img src="https://render.githubusercontent.com/render/math?math=N_m = \sum_{n = 1}^N z_{n, m} (5)">
</center>

_Main source:_ http://blog.manfredas.com/expectation-maximization-tutorial/

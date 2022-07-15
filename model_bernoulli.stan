data {
  // training data
  int<lower=1> C;                       // num categories
  int<lower=1> K;                       // num features
  int<lower=1> N;                       // num samples
  array[N] int<lower=1, upper=C> y;     // category of sample n
  array[N, K] int<lower=0, upper=1> x;  // sample n
  // hyperparameters
  vector<lower=0>[C] alpha;        // category prior
  array[K, 2] real<lower=0> beta;  // feature prior
}
parameters {
  simplex[C] theta;                        // category prevalence
  array[C, K] real<lower=0, upper=1> phi;  // feature parameter distribution
}
model {
  // priors
  theta ~ dirichlet(alpha);
  // we define a beta prior for phi for every feature and every class
  for (c in 1:C) {
    for (k in 1:K) {
      phi[c, k] ~ beta(beta[k, 1], beta[k, 2]);
    }
  }

  // data
  for (n in 1:N) {
    y[n] ~ categorical(theta);
  }
  // loop over every sample and every feature to let them
  // follow the distribution defined by the phi for that class
  for (n in 1:N) {
    for (k in 1:K) {
      x[n, k] ~ bernoulli(phi[y[n], k]);
    }
  }
}

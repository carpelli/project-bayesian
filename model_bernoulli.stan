data {
  // training data
  int<lower=1> C;                 // num categories
  int<lower=1> K;                 // num features
  int<lower=1> N;                 // num samples
  int<lower=1, upper=C> y[N];     // category of sample n
  int<lower=0, upper=1> x[N, K];  // sample n
  // hyperparameters
  vector<lower=0>[C] alpha;       // category prior
  real<lower=0> beta[C, K];       // feature prior
}
parameters {
  simplex[C] theta;                  // category prevalence
  real<lower=0, upper=1> phi[C, K];  // feature parameter distribution
}
model {
  theta ~ dirichlet(alpha);
  for (c in 1:C) {
    for (k in 1:K) {
      phi[c, k] ~ beta(beta[c, k], beta[c, k]);
    }
  }
  for (n in 1:N) {
    y[n] ~ categorical(theta);
  }
  for (n in 1:N) {
    for (k in 1:K) {
      x[n, k] ~ bernoulli(phi[y[n], k]);
    }
  }
}

data {
  // training data
  int<lower=1> C;             // num categories
  int<lower=1> K;             // num features
  int<lower=1> N;             // num samples
  int<lower=1, upper=C> y[N]; // category of sample n
  real x[N, K];               // sample n
  // hyperparameters
  vector<lower=0>[C] alpha;   // category prior
  real mu_mu;                 // feature mean prior
  real<lower=0> mu_sigma;     // feature mean prior
  real<lower=0> sigma_alpha;  // feature sd prior
  real<lower=0> sigma_beta;   // feature sd prior
}
parameters {
  simplex[C] theta;                  // category prevalence
  real mu[C, K];  // feature parameter distribution
  real<lower=0> sigma[C, K];  // feature parameter distribution
}
transformed parameters {
  real<lower=0> sigma_squared[C, K];
  for (c in 1:C) {
    for (k in 1:K) {
      sigma_squared[c, k] = sigma[c, k]^2;
    }
  }
}
model {
  theta ~ dirichlet(alpha);
  for (c in 1:C) {
    for (k in 1:K) {
      mu[c, k] ~ normal(mu_mu, mu_sigma);
      if (sigma_alpha > 0 && sigma_beta > 0)
        sigma_squared[c, k] ~ inv_gamma(sigma_alpha, sigma_beta);
    }
  }
  for (n in 1:N) {
    y[n] ~ categorical(theta);
  }
  for (n in 1:N) {
    for (k in 1:K) {
      x[n, k] ~ normal(mu[y[n], k], sigma[y[n], k]);
    }
  }
}

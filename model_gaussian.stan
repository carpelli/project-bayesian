data {
  // training data
  int<lower=1> C;                   // num categories
  int<lower=1> K;                   // num features
  int<lower=1> N;                   // num samples
  array[N] int<lower=1, upper=C> y; // category of sample n
  array[N, K] real x;                     // sample n
  // hyperparameters
  vector<lower=0>[C] alpha;   // category prior
  real mu_mu;                 // feature mean prior
  real<lower=0> mu_sigma;     // feature mean prior
  real<lower=0> sigma_alpha;  // feature sd prior
  real<lower=0> sigma_beta;   // feature sd prior
}
parameters {
  simplex[C] theta;                 // category prevalence
  array[C, K] real mu;              // feature mean distribution
  array[C, K] real<lower=0> sigma;  // feature sd distribution
}
transformed parameters {
  array[C, K] real <lower=0> sigma_squared;
  for (c in 1:C) {
    for (k in 1:K) {
      sigma_squared[c, k] = sigma[c, k]^2;
    }
  }
}
model {
  // priors
  theta ~ dirichlet(alpha);
  for (c in 1:C) {
    for (k in 1:K) {
      mu[c, k] ~ normal(mu_mu, mu_sigma);
      if (sigma_alpha > 0 && sigma_beta > 0) // if these values are zero we don't specify a prior
        sigma_squared[c, k] ~ inv_gamma(sigma_alpha, sigma_beta);
    }
  }

  //data
  for (n in 1:N) {
    y[n] ~ categorical(theta);
  }
  for (n in 1:N) {
    for (k in 1:K) {
      x[n, k] ~ normal(mu[y[n], k], sigma[y[n], k]);
    }
  }
}

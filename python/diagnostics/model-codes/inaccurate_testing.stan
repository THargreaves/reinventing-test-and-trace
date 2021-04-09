data {
  int<lower=0> N;                            // number of test observations
  int<lower=0> P;                            // number of places
  int<lower=0, upper=1> X[N,P];              // activity occurrences of tested individuals
  int<lower=0, upper=1> y[N];                // transmission (tested positive)
  real<lower=0, upper=1> mean_lambda[2];     // mean TP and TN test rates (for strong priors on tests)
  real<lower=0> se_lambda[2];                // standard error of test rates (for strong priors on tests)
}
parameters {
  real<lower=0, upper=1> theta[P];           // transmission rates
  real<lower=0, upper=1> rho;                // underlying risk
  real<lower=0, upper=1> lambda[2];          // True positive and true negative rates of tests [TP,TN]
}
model {
  // Precomputation
  real log1m_theta[P];
  real log1m_rho;
  real log_lambda[2];
  real log1m_lambda[2];
  real a_lambda[2];
  real b_lambda[2];
  
  for (p in 1:P) {
    log1m_theta[p] = log1m(theta[p]);
  }
  for(i in 1:2){
    log_lambda[i] = log(lambda[i]);
    log1m_lambda[i] = log1m(lambda[i]);
    
    a_lambda[i] = (((1-mean_lambda[i])/se_lambda[i]^2)-(1/mean_lambda[i]))*(mean_lambda[i]^2);
    b_lambda[i] = a_lambda[i]*((1/mean_lambda[i])-1);
  }
  
  log1m_rho = log1m(rho);
  
  // Priors
  theta ~ uniform(0, 1);
  rho ~ uniform(0, 1);
  lambda[1] ~ beta(a_lambda[1], b_lambda[1]);
  lambda[2] ~ beta(a_lambda[2], b_lambda[2]);

  // Likelihood
  for (n in 1:N) {
    real s = 0.0;
    for (p in 1:P) {
      if (X[n,p] == 1) {
        s += log1m_theta[p];
      }
    }
    s += log1m_rho;
    
    if (y[n] == 1) {
      target += log_sum_exp((log1m_exp(s) + log_lambda[1]), (s + log1m_lambda[2]));
    } 
    else {
      target += log_sum_exp((s + log_lambda[2]), (log1m_exp(s) + log1m_lambda[1]));
    }
  }
}
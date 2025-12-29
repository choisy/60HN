data {
  int<lower=1> N;                                 // number of observations
  int<lower=1> N_hosp;                            // number of hospitals
  int<lower=1> N_ward;                            // number of wards
  int<lower=1,upper=N_ward> ward_id[N];           // ward ID of each observation
  int<lower=1,upper=N_hosp> hosp_of_ward[N_ward]; // hospital ID of each ward
  real<lower=0> t[N];
  int<lower=1,upper=2> s0[N];
  int<lower=1,upper=2> s1[N];
}


parameters {
  // fixed intercepts (on log scale)
  real beta0_lambda; // log baseline rate for 1->2
  real beta0_mu;     // log baseline rate for 2->1

  // non-centered hospital effect on lambda
  vector[N_hosp] z_hosp_lambda;
  real<lower=0> sigma_hosp_lambda;

  // ward-level non-centered deviations around hospital effect on lambda
  vector[N_ward] z_ward_lambda;
  real<lower=0> sigma_ward_lambda;

  // non-centered hospital effect on mu
  vector[N_hosp] z_hosp_mu;
  real<lower=0> sigma_hosp_mu;
  
  // ward-level non-centered deviations around hospital effect on mu
  vector[N_ward] z_ward_mu;
  real<lower=0> sigma_ward_mu;
}


transformed parameters {
  vector[N_hosp] u_hosp_lambda = z_hosp_lambda * sigma_hosp_lambda;
  vector[N_ward] u_ward_lambda; // will be hospital-centered

  vector[N_hosp] u_hosp_mu = z_hosp_mu * sigma_hosp_mu;
  vector[N_ward] u_ward_mu;

  // make ward effects nested: ward effect = hospital effect for that ward + ward-specific deviation
  for (w in 1:N_ward) {
    u_ward_lambda[w] = u_hosp_lambda[hosp_of_ward[w]] + z_ward_lambda[w] * sigma_ward_lambda;
    u_ward_mu[w]     = u_hosp_mu[hosp_of_ward[w]]     + z_ward_mu[w] * sigma_ward_mu;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
transformed parameters {
  // actual random effects (hierarchical) - non-centered -> centered
  vector[N_hosp] u_hosp_lambda = z_hosp_lambda * sigma_hosp_lambda;
  vector[N_ward] u_ward_lambda = z_ward_lambda * sigma_ward_lambda;

  vector[N_hosp] u_hosp_mu = z_hosp_mu * sigma_hosp_mu;
  vector[N_ward] u_ward_mu = z_ward_mu * sigma_ward_mu;
}

lambda_i = exp(beta0_lambda + u_hosp_lambda[hosp_id[i]] + u_ward_lambda[ward_id[i]]);
lambda_i = exp(beta0_lambda + u_hosp_lambda[hosp_of_ward[ward[i]]] + z_ward_lambda[ward[i]] * sigma_ward_lambda;
///////////////////////////////////////////////////////////////////////////////////////

model {
  // priors
  beta0_lambda ~ normal(0, 2.5);
  beta0_mu ~ normal(0, 2.5);

  z_hosp_lambda ~ normal(0,1);
  z_ward_lambda ~ normal(0,1);
  sigma_hosp_lambda ~ normal(0,1);
  sigma_ward_lambda ~ normal(0,1);

  z_hosp_mu ~ normal(0,1);
  z_ward_mu ~ normal(0,1);
  sigma_hosp_mu ~ normal(0,1);
  sigma_ward_mu ~ normal(0,1);

  // likelihood (same as before, using u_hosp and u_ward)
  for (i in 1:N) {
    real lambda_i = exp(beta0_lambda + u_ward_lambda[ward_id[i]]); // hosp part already in u_ward
    real mu_i     = exp(beta0_mu + u_ward_mu[ward_id[i]]);
    real q = lambda_i + mu_i;
    real pi1 = mu_i / q;
    real pi2 = lambda_i / q;
    real e = exp(-q * t[i]);
    real p;
    if (s0[i] == 1) {
      if (s1[i] == 1) p = pi1 + pi2 * e;
      else p = pi2 * (1 - e);
    } else {
      if (s1[i] == 2) p = pi2 + pi1 * e;
      else p = pi1 * (1 - e);
    }
    target += log(fmax(p, 1e-12));
  }
}

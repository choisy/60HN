data {
  int<lower=1> N;                 // number of observations (patients)
  int<lower=1> N_hosp;            // number of hospitals
  int<lower=1> N_ward;            // number of wards
  int<lower=1,upper=N_ward> ward_id[N];  // ward for each observation
  int<lower=1,upper=N_hosp> hosp_id[N];  // hospital for each observation
  real<lower=0> t[N];             // follow-up time (discharge time - admission time)
  int<lower=1,upper=2> s0[N];     // state at admission (1 or 2)
  int<lower=1,upper=2> s1[N];     // state at discharge (1 or 2)
  // OPTIONAL: add patient-level covariates here if desired (not included in this template)
}


parameters {
  // fixed intercepts (on log scale)
  real beta0_lambda; // log baseline rate for 1->2
  real beta0_mu;     // log baseline rate for 2->1

  // non-centered hospital effects (lambda)
  vector[N_hosp] z_hosp_lambda;
  real<lower=0> sigma_hosp_lambda;

  // non-centered ward effects (lambda)
  vector[N_ward] z_ward_lambda;
  real<lower=0> sigma_ward_lambda;

  // non-centered hospital effects (mu)
  vector[N_hosp] z_hosp_mu;
  real<lower=0> sigma_hosp_mu;

  // non-centered ward effects (mu)
  vector[N_ward] z_ward_mu;
  real<lower=0> sigma_ward_mu;
}


transformed parameters {
  // actual random effects (hierarchical) - non-centered -> centered
  vector[N_hosp] u_hosp_lambda = z_hosp_lambda * sigma_hosp_lambda;
  vector[N_ward] u_ward_lambda = z_ward_lambda * sigma_ward_lambda;

  vector[N_hosp] u_hosp_mu = z_hosp_mu * sigma_hosp_mu;
  vector[N_ward] u_ward_mu = z_ward_mu * sigma_ward_mu;
}


model {
  // Priors
  beta0_lambda ~ normal(0, 2.5);
  beta0_mu     ~ normal(0, 2.5);

  z_hosp_lambda ~ normal(0, 1);
  sigma_hosp_lambda ~ normal(0, 1) T[0, ];
  z_ward_lambda ~ normal(0, 1);
  sigma_ward_lambda ~ normal(0, 1) T[0, ];

  z_hosp_mu ~ normal(0, 1);
  sigma_hosp_mu ~ normal(0, 1) T[0, ];
  z_ward_mu ~ normal(0, 1);
  sigma_ward_mu ~ normal(0, 1) T[0, ];

  // Likelihood
  for (i in 1:N) {
    // compute individual rates (positive)
    real lambda_i = exp(beta0_lambda + u_hosp_lambda[hosp_id[i]] + u_ward_lambda[ward_id[i]]);
    real mu_i     = exp(beta0_mu     + u_hosp_mu[hosp_id[i]]     + u_ward_mu[ward_id[i]]);

    real q = lambda_i + mu_i;
    // stationary probabilities
    real pi1 = mu_i / q; // prob in state 1 at equilibrium
    real pi2 = lambda_i / q;

    real e = exp(-q * t[i]);

    real p; // probability of observing s1 given s0 and t
    if (s0[i] == 1) {
      if (s1[i] == 1) {
        p = pi1 + pi2 * e;                // P11
      } else {
        p = pi2 * (1 - e);                // P12
      }
    } else { // s0 == 2
      if (s1[i] == 2) {
        p = pi2 + pi1 * e;                // P22
      } else {
        p = pi1 * (1 - e);                // P21
      }
    }
    // numerical safety: p should be in (0,1]
    p = fmax(p, 1e-12);
    target += log(p);
  }
}


generated quantities {
  // you can compute population-level average rates etc.
  real lambda_pop = exp(beta0_lambda);
  real mu_pop     = exp(beta0_mu);
}

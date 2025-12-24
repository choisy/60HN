data {
  int<lower = 1> N;                // number of transitions (adjacent observations)
  int<lower = 1> N_subj;           // number of subjects
  int<lower = 1> N_hosp;           // number of hospitals
  int<lower = 1> N_ward;           // number of wards (unique)
  int<lower = 1, upper = 2> s0[N]; // previous state (1 or 2)
  int<lower = 1, upper = 2> s1[N]; // next state (1 or 2)
  int<lower = 1> subj[N];          // subject index for each transition
  int<lower = 1> hosp[N];          // hospital index
  int<lower = 1> ward[N];          // ward index (nested)
  vector<lower = 0>[N] dt;         // time interval between observations
}


parameters {
  real log_q12_pop;                // population log intensity 1->2
  real log_q21_pop;                // population log intensity 2->1

  vector[N_subj] b_subj;           // subject random intercepts
  vector[N_hosp] u_hosp;           // hospital random effects
  vector[N_ward] v_ward;           // ward random effects

  real<lower=0> sigma_subj;
  real<lower=0> sigma_hosp;
  real<lower=0> sigma_ward;
}


transformed parameters {
// no transformation here; we'll exponentiate when making Q
}


model {
// priors
  log_q12_pop ~ normal(-1.2, 1.0); // weakly informative prior (log-rate)
  log_q21_pop ~ normal(-2.0, 1.0);

  sigma_subj ~ normal(0, 1);
  sigma_hosp ~ normal(0, 1);
  sigma_ward ~ normal(0, 1);

  b_subj ~ normal(0, sigma_subj);
  u_hosp ~ normal(0, sigma_hosp);
  v_ward ~ normal(0, sigma_ward);

// likelihood: for each observed adjacent pair, compute transition prob matrix and
// increment log-likelihood by the probability of seeing s1 given s0
  for (n in 1:N) {
    real q12 = exp(log_q12_pop + b_subj[subj[n]] + u_hosp[hosp[n]] + v_ward[ward[n]]);
    real q21 = exp(log_q21_pop + b_subj[subj[n]] + u_hosp[hosp[n]] + v_ward[ward[n]]);
    matrix[2,2] Q;
    matrix[2,2] P;
    Q[1,1] = -q12; Q[1,2] =  q12;
    Q[2,1] =  q21; Q[2,2] = -q21;
    P = matrix_exp(Q * dt[n]);            // Stan's matrix_exp (Stan Math)
// prevent numerical issues: force small positive lower bound
    vector[2] probs = to_vector(P[s0[n], ]);
    for (i in 1:2) {
      probs[i] = fmax(probs[i], 1e-12);
    }
    probs = probs / sum(probs);
    s1[n] ~ categorical(probs);
  }
}


generated quantities {
// empty for now; you can add posterior predictive draws if desired
}

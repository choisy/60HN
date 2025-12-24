data {
  int<lower=1> N_obs;               // total number of observations (patients * 2)
  int<lower=1> N_pat;               // number of patients
  int<lower=1> N_ward;              // total number of unique wards (across all hospitals)
  int<lower=1> N_hosp;              // total number of hospitals (should be 12 in your setup)
  int<lower=1> N_prov;              // number of provinces (2)

  int<lower=1,upper=N_pat> patient_id[N_obs]; // mapping observation -> patient
  int<lower=0,upper=1> y[N_obs];               // binary outcome (0/1) measured twice per patient

  // patient -> ward -> hospital -> province mappings:
  int<lower=1,upper=N_ward> ward_of_patient[N_pat];   // for each patient, which ward (1..N_ward)
  int<lower=1,upper=N_hosp> hosp_of_ward[N_ward];     // for each ward, which hospital (1..N_hosp)
  int<lower=1,upper=N_prov> prov_of_hosp[N_hosp];     // for each hospital, which province (1..N_prov)
}

parameters {
  real mu;                              // overall intercept (population log-odds)
  // non-centered parameters for multi-level intercepts:
  vector[N_prov] z_prov;
  vector[N_hosp] z_hosp;
  vector[N_ward] z_ward;
  vector[N_pat] z_pat;

  real<lower=0> sigma_prov;
  real<lower=0> sigma_hosp;
  real<lower=0> sigma_ward;
  real<lower=0> sigma_pat;
}

transformed parameters {
  vector[N_prov] a_prov = z_prov * sigma_prov;
  vector[N_hosp] a_hosp;
  vector[N_ward] a_ward;
  vector[N_pat] a_pat;

  // build hospital effects by mapping hospital -> its province effect
  // Note: hospital-level effect is modeled independent but we keep province mapping
  for (h in 1:N_hosp) {
    a_hosp[h] = z_hosp[h] * sigma_hosp;
  }
  for (w in 1:N_ward) {
    a_ward[w] = z_ward[w] * sigma_ward;
  }
  for (p in 1:N_pat) {
    a_pat[p] = z_pat[p] * sigma_pat;
  }
}

model {
  // Priors
  mu ~ normal(0, 2);               // weakly informative prior for log-odds
  z_prov ~ normal(0,1);
  z_hosp ~ normal(0,1);
  z_ward ~ normal(0,1);
  z_pat ~ normal(0,1);

  sigma_prov ~ normal(0, 1) T[0,];
  sigma_hosp ~ normal(0, 1) T[0,];
  sigma_ward ~ normal(0, 1) T[0,];
  sigma_pat ~ normal(0, 1) T[0,];

  // Likelihood: each observation belongs to a patient; patient has patient-level effect,
  // whose ward/hospital/province are known via mappings
  for (i in 1:N_obs) {
    int pid = patient_id[i];
    int wid = ward_of_patient[pid];
    int hid = hosp_of_ward[wid];
    int prid = prov_of_hosp[hid];

    real lp = mu
              + a_prov[prid]   // province intercept
              + a_hosp[hid]    // hospital intercept
              + a_ward[wid]    // ward intercept
              + a_pat[pid];    // patient intercept

    y[i] ~ bernoulli_logit(lp);
  }
}

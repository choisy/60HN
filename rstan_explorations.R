library(rstan)
#library(tidyverse)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

set.seed(123)

# -------------------------
# 1. Simulate data
# -------------------------
n_hosp <- 4
wards_per_hosp <- 3
n_subjects <- 60
obs_per_subject <- 6
n_wards <- n_hosp * wards_per_hosp

# true (population) baseline log intensities
log_q12_pop <- log(0.3)   # baseline rate 1->2
log_q21_pop <- log(0.15)  # baseline rate 2->1

# true sd of random effects
sd_hosp <- 0.4
sd_ward <- 0.3
sd_subj <- 0.6

# draw random effects
u_hosp <- rnorm(n_hosp, 0, sd_hosp)
v_ward <- rnorm(n_wards, 0, sd_ward)
b_subj <- rnorm(n_subjects, 0, sd_subj)

sample_wR <- function(...) sample(..., replace = TRUE)
rstat <- function(...) sample(1:2, 1, ...)

df <- tibble(hosp = sample_wR(n_hosp, n_subjects)) |> 
  group_by(hosp) |> 
  mutate(ward = paste0(hosp, sample_wR(wards_per_hosp, n()))) |> 
  ungroup() |> 
  mutate(across(ward, ~ as.integer(as.factor(as.numeric(.x))))) |> 
  arrange(hosp, ward) |> 
  mutate(subj = seq_len(n_subjects)) |> 
  group_by(subj) |> 
  mutate(time = list(c(0, cumsum(rexp(obs_per_subject - 1, .2))))) |> 
  unnest(time) |>
  ungroup() |> 
  mutate(var = b_subj[subj] + u_hosp[hosp] + v_ward[ward],
         q12 = exp(log_q12_pop + var),
         q21 = exp(log_q21_pop + var)) |> 
  rowwise() |> 
  mutate(Q = list(matrix(c(-q12, q21, q12, -q21), 2))) |> 
  group_by(subj) |> 
  mutate(P = map2(Q, c(0, diff(time)), ~ expm::expm(.x * .y)),
         state = accumulate(tail(P, -1), ~ rstat(prob = .y[.x, ]), .init = rstat())) |> 
  ungroup() |> 
  select(subj, hosp, ward, time, state)


# create data in "adjacent-observation" form:
# for each adjacent pair (t_{k-1} -> t_k) we need prev_state, next_state, dt, and indices
pairs <- df |> 
  group_by(subj) |> 
  mutate(prev_state = lag(state),
         prev_time  = lag(time)) |> 
  ungroup() |> 
  na.exclude() |> 
  rename(next_state = state) |> 
  mutate(N = row_number(),
         dt = time - prev_time) |> 
  select(-time, -prev_time)
  
# convert to vectors for Stan
N_obs <- nrow(pairs)
subj_vec <- pairs$subj
hosp_vec <- pairs$hosp
ward_vec <- pairs$ward
s0 <- pairs$prev_state
s1 <- pairs$next_state
dt_vec <- pairs$dt

# -------------------------
# 2. Stan model string
# -------------------------
stan_code <- "
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
"

# compile Stan model
sm <- stan_model(model_code = stan_code)

# prepare data list for Stan
stan_dat <- list(
  N = N_obs,
  N_subj = n_subjects,
  N_hosp = n_hosp,
  N_ward = n_wards,
  s0 = s0,
  s1 = s1,
  subj = subj_vec,
  hosp = hosp_vec,
  ward = ward_vec,
  dt = dt_vec
)

# -------------------------
# 3. Fit model (sampling)
# -------------------------
fit <- sampling(sm, data = stan_dat, iter = 1200, warmup = 600, chains = 4,
                control = list(adapt_delta = 0.9, max_treedepth = 12))

print(fit, pars = c("log_q12_pop", "log_q21_pop", "sigma_subj", "sigma_hosp", "sigma_ward"),
      probs = c(0.025, 0.5, 0.975))

# -------------------------
# Notes:
# - This is a toy example. For real data, consider identifiability and prior choices.
# - The model places the same random effect additively on both log(q12) and log(q21).
#   You can modify to allow different random effects per transition if desired.
# -------------------------


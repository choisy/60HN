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
n_wards <- n_hosp * wards_per_hosp

n_subjects <- 60
obs_per_subject <- 6

# make subject -> hospital & ward mapping
subject_hosp <- sample(1:n_hosp, n_subjects, replace = TRUE)
# assign wards nested within each hospital: ward index = (hosp-1) * wards_per_hosp + 1..wards_per_hosp
subject_ward <- sapply(subject_hosp, function(h) {
  (h - 1) * wards_per_hosp + sample(1:wards_per_hosp, 1)
})


sample_wR <- function(...) sample(..., replace = TRUE)

tibble(hosp = sample_wR(n_hosp, n_subjects)) |> 
  group_by(hosp) |> 
  mutate(ward = paste0(hosp, sample_wR(wards_per_hosp, n()))) |> 
  ungroup() |> 
  mutate(across(ward, ~ as.integer(as.factor(as.numeric(.x))))) |> 
  arrange(hosp, ward) |> 
  mutate(subj = seq_len(n_subjects)) |> 
  select(subj, everything()) |> 
  group_by(subj) |> 
  
  


  
  

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

# simulate observation times and states
rows <- list()
for (i in 1:n_subjects) {
  times <- sort(c(0, cumsum(rexp(obs_per_subject - 1, rate = 0.2)))) # irregular times
  state <- integer(length(times))
  # initial state randomly 1 or 2
  state[1] <- sample(1:2, 1)
  for (k in 2:length(times)) {
    dt <- times[k] - times[k-1]
    # subject-specific intensities (multiply on original scale via additive on log)
    q12_i <- exp(log_q12_pop + b_subj[i] + u_hosp[subject_hosp[i]] + v_ward[subject_ward[i]])
    q21_i <- exp(log_q21_pop + b_subj[i] + u_hosp[subject_hosp[i]] + v_ward[subject_ward[i]])
    # build Q-matrix
    Q <- matrix(c(-q12_i, q12_i,
                  q21_i, -q21_i), nrow = 2, byrow = TRUE)
    # compute transition probability matrix P = expm(Q * dt)
    P <- expm::expm(Q * dt)
    # sample next state based on P
    state[k] <- sample(1:2, 1, prob = P[state[k-1], ])
  }
  rows[[i]] <- data.frame(
    subj = i,
    hosp = subject_hosp[i],
    ward = subject_ward[i],
    time = times,
    state = state
  )
}
df <- bind_rows(rows) %>% arrange(subj, time) %>% mutate(state = as.integer(state))

# create data in "adjacent-observation" form:
# for each adjacent pair (t_{k-1} -> t_k) we need prev_state, next_state, dt, and indices
pairs <- df %>%
  group_by(subj) %>%
  arrange(time) %>%
  mutate(prev_state = lag(state),
         prev_time = lag(time)) %>%
  filter(!is.na(prev_state)) %>%
  ungroup() %>%
  transmute(
    N = row_number(),
    subj = subj,
    hosp = hosp,
    ward = ward,
    prev_state = prev_state,
    next_state = state,
    dt = time - prev_time
  ) %>% mutate_all(as.integer)

# convert to vectors for Stan
N_obs <- nrow(pairs)
subj_vec <- pairs$subj
hosp_vec <- pairs$hosp
ward_vec <- pairs$ward
s0 <- pairs$prev_state
s1 <- pairs$next_state
dt_vec <- as.numeric(pairs$dt)

# -------------------------
# 2. Stan model string
# -------------------------
stan_code <- "
data {
  int<lower=1> N;                 // number of transitions (adjacent observations)
  int<lower=1> N_subj;           // number of subjects
  int<lower=1> N_hosp;           // number of hospitals
  int<lower=1> N_ward;           // number of wards (unique)
  int<lower=1,upper=2> s0[N];    // previous state (1 or 2)
  int<lower=1,upper=2> s1[N];    // next state (1 or 2)
  int<lower=1> subj[N];          // subject index for each transition
  int<lower=1> hosp[N];          // hospital index
  int<lower=1> ward[N];          // ward index (nested)
  vector<lower=0>[N] dt;         // time interval between observations
}
parameters {
  real log_q12_pop;              // population log intensity 1->2
  real log_q21_pop;              // population log intensity 2->1

  vector[N_subj] b_subj;         // subject random intercepts
  vector[N_hosp] u_hosp;         // hospital random effects
  vector[N_ward] v_ward;         // ward random effects

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
    Q[1,1] = -q12; Q[1,2] = q12;
    Q[2,1] = q21;  Q[2,2] = -q21;
    P = matrix_exp(Q * dt[n]);            // Stan's matrix_exp (Stan Math)
    // prevent numerical issues: force small positive lower bound
    vector[2] probs = colwise_max(to_vector(P[s0[n], ]), rep_vector(1e-12, 2));
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



#######################################################################################


df |> 
  select(subj, hosp, ward) |> 
  unique() |> 
  group_by(hosp, ward) |> 
  tally()

df |> 
  select(hosp, ward) |> 
  arrange(hosp, ward) |> 
  unique() |> 
  group_by(hosp) |> 
  tally()

library(tidyverse)
library(haven)
library(DBI)
library(dbplyr)
library(softImpute)
library(MASS)

path_to_dbname <- "~/data/nhanes/nhanes_012324.sqlite"

con <- DBI::dbConnect(RSQLite::SQLite(), dbname=path_to_dbname)
DBI::dbListTables(con)

# List of variables to include: metals, dietary variables (DRXTOT_B Doc), Cotinine and Hydroxycotinine - Serum, Fatty Acids - Serum, Folate, Hep A-E, Mercury (both), Physical Activity	

var_names <- tbl(con, "variable_names_epcf")
vars <- var_names %>% collect() %>% filter(Begin.Year == 2013) %>% filter(Data.File.Name == "DR1TOT_H")

# Tables to grab from 2013-14 survey

data_h <- data.frame()

demo_h <- tbl(con, "DEMO_H")
demo_h_tbl <- demo_h %>% collect()

diet_h <- tbl(con, "DR2TOT_H")
diet_h_tbl <- diet_h %>% collect()
diet_h_tbl <- diet_h_tbl %>% dplyr::select(SEQN, DR2TKCAL:DR2TP226) %>% na.omit()

data_h <- right_join(demo_h_tbl, diet_h_tbl)

cotinine_h <- tbl(con, "COT_H")
cotinine_h_tbl <- cotinine_h %>% collect() # 9 variables
cotinine_h_tbl <- cotinine_h_tbl %>% na.omit()

data_h <- inner_join(data_h, cotinine_h_tbl)

folate_h <- tbl(con, "FOLFMS_H")
folate_h_tbl <- folate_h %>% collect()
folate_h_tbl <- folate_h_tbl %>% na.omit()

data_h <- inner_join(data_h, cotinine_h_tbl)

# metals_h <- tbl(con, "UM_H")
# metals_h_tbl <- metals_h %>% collect()

# mercury_urine_h <- tbl(con, "UHG_H")
# mercury_urine_h_tbl <- mercury_h %>% collect()

# mercury_blood_h <- tbl(con, "IHGEM_H")
# mercury_blood_h_tbl <- mercury_blood_h %>% collect()

physical_activity_h <- tbl(con, "PAQ_H")
physical_activity_h_tbl <- physical_activity_h %>% collect()
physical_activity_h_tbl <- physical_activity_h_tbl %>% dplyr::select(SEQN, PAQ_work_vigorous_week:PAQ_total_met) %>% na.omit()
data_h <- inner_join(data_h, physical_activity_h_tbl)


data_h <- data_h[, colSums(is.na(data_h)) == 0]

n_variables <- 100
mean_vector <- rep(0, n_variables)
random_matrix <- matrix(rnorm(n_variables^2), n_variables, n_variables)

# Ensure the matrix is positive definite
covariance_matrix <- crossprod(random_matrix)
scaling_factors <- runif(n_variables, 0.5, 1.5)
covariance_matrix <- diag(scaling_factors) %*% covariance_matrix %*% diag(scaling_factors)
simulated_data <- mvrnorm(n = n_observations, mu = mean_vector, Sigma = covariance_matrix)


data_h <- cbind(data_h, simulated_data)


# Data from the g survey

# Tables to grab from 2013-14 survey

data_g <- data.frame()

demo_g <- tbl(con, "DEMO_G")
demo_g_tbl <- demo_g %>% collect()

diet_g <- tbl(con, "DR2TOT_G")
diet_g_tbl <- diet_g %>% collect()
diet_g_tbl <- diet_g_tbl %>% dplyr::select(SEQN, DR2TKCAL:DR2TP226) %>% na.omit()

data_g <- right_join(demo_g_tbl, diet_g_tbl)

cotinine_g <- tbl(con, "COTNAL_G")
cotinine_g_tbl <- cotinine_g %>% collect() # 9 variables
cotinine_g_tbl <- cotinine_g_tbl %>% na.omit()

data_g <- inner_join(data_g, cotinine_g_tbl)

folate_g <- tbl(con, "FOLFMS_G")
folate_g_tbl <- folate_g %>% collect()
folate_g_tbl <- folate_g_tbl %>% na.omit()

data_g <- inner_join(data_g, cotinine_g_tbl)

# metals_h <- tbl(con, "UM_H")
# metals_h_tbl <- metals_h %>% collect()

# mercury_urine_h <- tbl(con, "UHG_H")
# mercury_urine_h_tbl <- mercury_h %>% collect()

# mercury_blood_h <- tbl(con, "IHGEM_H")
# mercury_blood_h_tbl <- mercury_blood_h %>% collect()

physical_activity_g <- tbl(con, "PAQ_G")
physical_activity_g_tbl <- physical_activity_g %>% collect()
physical_activity_g_tbl <- physical_activity_g_tbl %>% dplyr::select(SEQN, PAQ_work_vigorous_week:PAQ_total_met) %>% na.omit()
data_g <- inner_join(data_g, physical_activity_g_tbl)


data_g <- data_g[, colSums(is.na(data_g)) == 0]

n_variables <- 100
mean_vector <- rep(0, n_variables)
random_matrix <- matrix(rnorm(n_variables^2), n_variables, n_variables)

# Ensure the matrix is positive definite
covariance_matrix <- crossprod(random_matrix)
scaling_factors <- runif(n_variables, 0.5, 1.5)
covariance_matrix <- diag(scaling_factors) %*% covariance_matrix %*% diag(scaling_factors)
simulated_data <- mvrnorm(n = n_observations, mu = mean_vector, Sigma = covariance_matrix)


data_h <- cbind(data_h, simulated_data)

colnames_g <- colnames(data_g)
colnames_h <- colnames(data_h)

cols_only_in_df1 <- setdiff(colnames_g, colnames_h)
cols_only_in_df2 <- setdiff(colnames_h, colnames_g)

data_g$URXNAL <- NULL
data_g$URDNALLC <- NULL
data_g$URXUCR <- NULL

data_h$LBXHCT <- NULL
data_h$LBDHCTLC <- NULL

full_data <- rbind(data_g, data_h)

n_observations <- nrow(full_data)
n_variables <- 100
mean_vector <- rep(0, n_variables)
random_matrix <- matrix(rnorm(n_variables^2), n_variables, n_variables)

# Ensure the matrix is positive definite
covariance_matrix <- crossprod(random_matrix)
scaling_factors <- runif(n_variables, 0.5, 1.5)
covariance_matrix <- diag(scaling_factors) %*% covariance_matrix %*% diag(scaling_factors)
simulated_data <- mvrnorm(n = n_observations, mu = mean_vector, Sigma = covariance_matrix)

full_data_sim <- cbind(full_data, simulated_data)
full_data_sim$SDDSRVYR <- NULL
full_data_sim$SDMVPSU <- NULL
full_data_sim$SDMVSTRA <- NULL
full_data_sim$SEQN <- NULL
full_data_sim$WTINT2YR <- NULL
full_data_sim$WTMEC2YR <- NULL
full_data_sim$AGE_SQUARED <- NULL

write.csv(full_data_sim, "~/data/nhanes/full_data_sim.csv")

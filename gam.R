library(mgcv)
library(MASS)
library(dplyr)
library(GGally)
library(fields)
library(splines)

corr_matrix <- diag(num_vars)

# Function to set random correlations
set_random_correlations <- function(corr_matrix, high_prob = 0.8, low_prob = 0.1, mid_prob = 0.1) {
  for (i in 1:(num_vars - 1)) {
    for (j in (i + 1):num_vars) {
      r <- runif(1)
      if (r < high_prob) {
        # Set high correlation
        corr_matrix[i, j] <- runif(1, 0.7, 0.9)
        corr_matrix[i,j] <- sample(c(-1, 1), 1, replace = TRUE)*corr_matrix[i,j] 
      } else if (r < high_prob + low_prob) {
        # Set low correlation
        corr_matrix[i, j] <- runif(1, 0.1, 0.3)
        corr_matrix[i,j] <- sample(c(-1, 1), 1, replace = TRUE)*corr_matrix[i,j] 
      } else {
        # Set intermediate correlation
        corr_matrix[i, j] <- runif(1, 0.3, 0.7)
        corr_matrix[i,j] <- sample(c(-1, 1), 1, replace = TRUE)*corr_matrix[i,j] 
      }
      corr_matrix[j, i] <- corr_matrix[i, j]
    }
  }
  return(corr_matrix)
}

# Set correlations in the matrix
corr_matrix <- set_random_correlations(corr_matrix)

# Check if the matrix is positive definite
is_pos_def <- all(eigen(corr_matrix)$values > 0)
is_pos_def
PD <- nearPD(corr_matrix, corr = TRUE)
all(eigen(PD$mat)$values > 0)
Sigma <- PD$mat

# Print a subset of the correlation matrix for inspection
print("Subset of the Correlation Matrix with Varying Correlations:")
print(Sigma[1:10, 1:10])
max(Sigma[upper.tri(Sigma)])

random_data <- mvrnorm(n = 1000, mu = rep(0, num_vars), Sigma = Sigma)

# Convert to a dataframe
random_data_df <- as.data.frame(random_data)


ggpairs(random_data_df[,1:10])

# Spline time

set.seed(42)

y_base <- rep(0, 1000)

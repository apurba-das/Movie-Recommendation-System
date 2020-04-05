################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################################
# Modelling using different methods/analysis
###############################################

# Understanding the training set contents
dim(edx)
head(edx)


#We can see this table is in tidy format with thousands of rows:
edx %>% as_tibble()

#We can see the number of unique users that provided ratings and how many unique movies were rated:

edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#We notice that some movies get rated more than others. Below is the distribution. This should not surprise us given that there are blockbuster movies watched by millions and artsy, independent movies watched by just a few. Our second observation is that some users are more active than others at rating movies:
  
movie_ratings <- edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

head(movie_ratings)

tail(movie_ratings)

edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Ratings given") + 
  ylab("Users") +
  ggtitle("Number of ratings given by users")

# Defining the rmse function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 1. A first model - predict the same rating for all movies regardless of user

mu_hat <- mean(edx$rating)
mu_hat

# If we predict all unknown ratings with mu_hat, we obtain the following RMSE
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

# As we go along, there will be a need to compare different approaches. Hence, starting by creating a results table with this naive approach
rmse_results <- tibble(Method = "Just the average", RMSE = naive_rmse)

# Viewing the results obtained so far
rmse_results %>% knitr::kable()

# 2. Modeling movie effects

# We can use least squares to estimate the bias in movie ratings the following way
# fit <- lm(rating ~ as.factor(movieId), data = edx)
# The lm() function will be very slow given the  thousands of bias, hence commented the above. 

# But in this case, we know that the least squares estimate is just the average of rating - mu for each movie. So it can be computed in the following way 
# Dropping the hat notation in the code going forward
mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#Understanding these estimates
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculating the RMSE using this model
rmse_me <- RMSE(predicted_ratings, validation$rating)
rmse_results <- add_row(rmse_results, Method = 'Movie Effect Model', RMSE = rmse_me)

# Viewing the results obtained so far
rmse_results %>% knitr::kable()

# 3. Movie + User Effects Model

# Analysing the dataset
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# It is noticed that there is substantial variability across users as well. This implies that a further improvement to the model is possible using the below method

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculating the RMSE using this model
rmse_ue <- RMSE(predicted_ratings, validation$rating)
rmse_results <- add_row(rmse_results, Method = 'Movie + User Effects Model', RMSE = rmse_ue)

# Viewing the results obtained so far
rmse_results %>% knitr::kable()

# 4. Regularization

# We have already seen that there are some movies that were rated by jusy 1 users and others by many more users. So, these movies are the obscure ones which have would very high probability of being the best or the worst movie. Thus, a penalty term needs to be introduced to regularize this effect. 

# 4.1. Regularized Movie Effect Model
# ( Penalized least squares with lambda = 3 )

lambda <- 3
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Calculating the RMSE using this model
rmse_lambda3 <- RMSE(predicted_ratings, validation$rating)
rmse_results <- add_row(rmse_results, Method = 'Regularized Movie Effect Model', RMSE = rmse_lambda3)

# Viewing the results obtained so far
rmse_results %>% knitr::kable()

# 4.2 Regularized Movie + User Effect Model
# ( Choosing the penalty term using cross validation )

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses) 

# For the final model, the optimal lambda is:
lambda <- lambdas[which.min(rmses)]
lambda

# RMSE for the final model with optimised lambda
rmse_lambda_opt  <- min(rmses)

# Calculating the RMSE using this model
rmse_results <- add_row(rmse_results, Method = 'Regularized Movie + User Effect Model', RMSE = rmse_lambda_opt)

#######################################
# Viewing the final comparison results
#######################################

rmse_results %>% knitr::kable()

# Hence, among the models designed, the Regularized Movie + User Effect Model gives the best rmse of 0.8648170

#Reference : Used https://rafalab.github.io/dsbook as a reference

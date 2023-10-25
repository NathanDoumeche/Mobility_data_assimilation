#' @title Offline prediction with discrimination
#' @description
#' This function uses a discriminating vector of values and learns offline. For any value 
#' of the discriminating list, it learns for the train pairs corresponding to this
#' particular value and then prediction is realized for both train and test pairs.
#' 
#' @param X matrix containing the explanatory variables
#' @param y time series to predict
#' @param train_set vector of indices of the training set
#' @param discr vector containing the discrimating values
#' @param params (optional, default NULL) list containing the initial values of theta and P
#' @param distribution (optional, default gaussian)
#' 
#' @return y_hat the offline discriminated prediction of y
#' @export
predict_offline_discriminate <- function(X,y,train_set,discr,params=NULL,
                                         distribution='gaussian') {
  prev <- numeric(length(y))
  d <- dim(X)[2]
  if (is.null(params))
    params <- list('theta'=matrix(0,d,1),'P'=diag(d))
  for (val in unique(discr)) {
    select <- train_set[which(discr[train_set] == val)]
    theta <- final_parameters(params,X[select,],y[select],distribution=distribution)$theta
    prev[discr==val] <- X[discr==val,] %*% theta
  }
  if (distribution == 'logistic')
    prev <- 1 / (1+exp(-prev))
  prev
}
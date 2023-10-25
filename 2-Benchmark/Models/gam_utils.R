#' @title Discriminated GAM
#' @description
#' A way to create a GAM discriminating with a parameter
#' (instant of the day for instance).
#' Ir relies on the mgcv implementation.
#'
#' @param formula the formula to feed \code{mgcv::gam}
#' @param data the data on which the models are trained
#' @param discr vector containing the discrimating values
#'
#' @return a list of GAM models.
#' @export
gam_discriminate <- function(formula, data, discr, ncores=1, select=FALSE, gamma=1) {
  model <- list()
  model$values <- unique(discr)
  model$gams <- parallel::mclapply(1:length(model$values), function(i) {
    mgcv::gam(formula, data = data[which(discr == model$values[i]),], select=select, gamma=gamma)
  },
  mc.cores=ncores)
  model
}


gamlss_discriminate <- function(list_formula, data, discr, ncores=1) {
  model <- list()
  model$values <- unique(discr)
  model$gams <- parallel::mclapply(1:length(model$values), function(i) {
    mgcv::gam(list_formula, data = data[which(discr == model$values[i]),], family=gaulss())
  },
  mc.cores=ncores)
  model
}




#' @title Prediction for a discriminated GAM
#' @description The function handles the model as defined by \code{gam_discriminate}
#'
#' @param model the model obtained from a call to \code{gam_discriminate}
#' @param newdata the data to make the prediction for
#' @param discr vector containing the discrimating values
#' @param type (optional, default response) type parameter passed to the GAM function
#'
#' @return the vector of predictions
#' @export
predict_gam_discriminate <- function(model, newdata, discr, type='response', gamlss=FALSE) {
  if (type == 'response') {
    prediction <- rep(NA, nrow(newdata))
    for (val in unique(discr))
    {
      if(gamlss==TRUE)
      {
      prediction[which(discr == val)] <-
        mgcv::predict.gam(model$gams[[which(model$values == val)]],
                          newdata[which(discr == val), ])[,1]
      }
      if(gamlss==FALSE)
      {
        prediction[which(discr == val)] <-
          mgcv::predict.gam(model$gams[[which(model$values == val)]],
                            newdata[which(discr == val), ])
      }
    }
    return(prediction)
  }
  d <- dim(mgcv::predict.gam(model$gams[[1]], newdata[1:2, ], type=type))[2]
  prediction <- matrix(0, nrow(newdata), d)
  for (val in unique(discr)) {
    prediction[which(discr == val),] <-
      mgcv::predict.gam(model$gams[[which(model$values == val)]],
                        newdata[which(discr == val), ], type=type)
  }
  prediction
}

#' @title Prepare data from GAM models
#' @description Prepare the data to feed to Kalman Filter. The GAM models are used to
#' create the non-linear transforms, and the KF uses the non-linear outputs to build a
#' linear transform on top of them. We can either use the effects (\code{type=terms})
#' or modify even the splines by using lower-level features (\code{type=pmatrix}).
#'
#' @param model the model obtained from a call to \code{gam_discriminate}
#' @param data the data to feed to the GAMs
#' @param discr vector containing the discrimating values
#' @param estim the training set on which we compute the mean and standard deviation of
#' each variable
#' @param cst (optional, default 1) a constant variable to get an intercept term
#' in the model
#' @param type (optional, default terms) type parameter passed to the GAM function
#'
#' @return the matrix containing the explanatory variables
#' @export
prep_data_gam <- function(model, data, discr, estim, cst=1, type='terms') {
  X <- predict_gam_discriminate(model, data, discr, type=type)
  params <- get_mean_sd_discriminate(X[estim,], discr[estim])
  X_standardize <- standardize_discriminate(X, discr, params, cst=cst)
  X_standardize
}

prep_data_gam_residuals <- function(model_1, model_2, data, discr, estim, cst=1, type='terms', common_terms="all") {
  X_1 <- predict_gam_discriminate(model_1, data, discr, type="terms")
  X_2 <- predict_gam_discriminate(model_2, data, discr, type="terms")
  
  X=X_1
  X[,common_terms] <- X_1[,common_terms] + X_2[,1:length(common_terms)]
  X <- cbind(X, X_2[,-c(1:length(common_terms))])
  
  params <- get_mean_sd_discriminate(X[estim,], discr[estim])
  X_standardize <- standardize_discriminate(X, discr, params, cst=cst)
  return(X_standardize)
}


#' @export
get_mean_sd <- function(X) {
  list(col_mean=colMeans(X),
       col_sd=sapply(1:ncol(X), function(j) sd(X[,j])))
}

#' @export
get_mean_sd_discriminate <- function(X, discr) {
  values <- unique(discr)
  COL_MEAN <- matrix(0, length(values), ncol(X))
  COL_SD <- matrix(0, length(values), ncol(X))
  for (i in 1:length(values)) {
    params <- get_mean_sd(X[which(discr == values[i]),])
    COL_MEAN[i,] <- params$col_mean
    COL_SD[i,] <- params$col_sd
  }
  list(COL_MEAN=COL_MEAN, COL_SD=COL_SD, values=values)
}

#' @export
standardize <- function(X, params, cst=1) {
  ans <- X
  for (j in 1:ncol(X)) {
    ans[,j] <- (X[,j] - params$col_mean[j]) / (if (params$col_sd[j] > 0) params$col_sd[j] else 1)
  }
  cbind(ans, cst)
}

#' @export
standardize_discriminate <- function(X, discr, params, cst=1) {
  ans <- cbind(X, cst)
  for (v in unique(discr)) {
    i <- which(params$values == v)[1]
    sel <- which(discr == v)
    ans[sel,] <- standardize(matrix(X[sel,], length(sel), ncol(X)),
                             list(col_mean=params$COL_MEAN[i,], col_sd=params$COL_SD[i,]),
                             cst=cst)
  }
  ans
}

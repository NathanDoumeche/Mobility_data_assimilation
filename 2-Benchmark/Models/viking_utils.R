require('viking')

static_discriminate <- function(X, y, discr, ncores=1) {
  model <- list()
  model$values <- unique(discr)
  model$ssm <- parallel::mclapply(1:length(model$values), function(i) {
    sel <- which(discr == model$values[i])
    viking::statespace( X[sel,], y[sel])
  },
  mc.cores=ncores)
  model
}

predict_static_discriminate <- function(model, discr) {
  prediction <- rep(NA, length(discr))
  for (val in unique(discr))
    prediction[which(discr == val)] <- model$ssm[[val+1]]$pred_mean
  return(prediction)
}



dynamic_discriminate <- function(ssm, X, y, discr, q_list = 2^(-30:0), p1 = 1, ncores=1, method='igd') {
  model <- list()
  model$values <- unique(discr)
  model$ssm_dyn <- parallel::mclapply(1:length(model$values), function(i) {
    sel <- which(discr == model$values[i])
    viking::select_Kalman_variances(ssm$ssm[[i]], X[sel,], y[sel], q_list=q_list, p1=p1, ncores=1, method=method)
  },
  mc.cores=ncores)
  model
}


viking_discriminate <- function(ssm_dyn, X, y, discr,  ncores=1, seed) {
  model <- list()
  model$values <- unique(discr)
  model$ssm <- parallel::mclapply(1:length(model$values), function(i) {
    set.seed(seed[i])
    sel <- which(discr == model$values[i])
    kalman_params <- ssm_dyn$ssm_dyn[[i]]$kalman_params
    viking::statespace(X[sel,], y[sel]/kalman_params$sig, viking_params = list(theta=kalman_params$theta/kalman_params$sig, P=kalman_params$P/kalman_params$sig^2,
                                                                               s=10^-2,
                                                                               hatb=exp(diag(kalman_params$Q/kalman_params$sig^2))-1, 
                                                                               Sigma=diag(dim(X)[2],x=10^-3),
                                                                               rho_a=0, rho_b=10^-5))
    
  },
  mc.cores=ncores)
  
  for(i in c(1:length(model$values)))
  {
    kalman_params <- ssm_dyn$ssm_dyn[[i]]$kalman_params
    model$ssm[[i]]$kalman_params <- kalman_params
  }
  
  model
}



# ssm_vik <- viking::statespace(X, y/l$sig, 
#                               viking_params = list(theta=l$theta/l$sig, P=l$P/l$sig^2,
#                                                    s=10^-2,
#                                                    hatb=exp(diag(l$Q/l$sig^2))-1, 
#                                                    Sigma=diag(d,x=10^-3),
#                                                    rho_a=0, rho_b=10^-5))
# plot(ssm_vik, pause=F, window_size = 14, date = data$Date, sel = test)
# prev$Viking <- ssm_vik$pred_mean * l$sigma




predict_viking_discriminate <- function(model, X, y, discr, model_type="dynamic") {
  prediction <- rep(NA, length(discr))
  
  if(model_type=="dynamic")
  {
    for (val in model$values)
      {
        sel <- which(discr == val)
        model$ssm_dyn[[val+1]] <- predict(model$ssm_dyn[[val+1]], X[sel,], y[sel], type='model', compute_smooth = TRUE)
        prediction[which(discr == val)] <- model$ssm_dyn[[val+1]]$pred_mean
        #print(val)

    }
  }
  
  if(model_type=="viking")
  {
    for (val in model$values)
    {
      sel <- which(discr == val)
      prediction[which(discr == val)] <- model$ssm[[val+1]]$pred_mean * model$ssm[[val+1]]$kalman_params$sig
    }
  }
  
  return(prediction)
}

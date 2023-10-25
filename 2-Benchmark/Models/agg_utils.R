agg_discriminate <- function(Y, experts, discr, ncores=1)
{
  agg <- list()
  agg$values <- unique(discr)
  
  agg$mixture <- parallel::mclapply(1:length(agg$values), function(i) {
    sel <- which(discr == agg$values[i])
    opera::mixture(Y[sel], experts[sel,])
  },
  mc.cores=ncores)
  
  return(agg)
}



predict_agg_discriminate <- function(agg, discr) {
  prediction <- rep(NA, length(discr))
    for (val in agg$values)
    {
      sel <- which(discr == val)
      prediction[which(discr == val)] <- agg$mixture[[val+1]]$prediction
    }
  return(prediction)
}


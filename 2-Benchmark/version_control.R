versions = data.frame(installed.packages())[c("Package", "Version", "Depends")]
saveRDS(versions, "Clean_codes/packages_versions.RDS")

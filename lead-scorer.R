

main = function() {
# df <- read.csv("data/leads-and-customers.csv")
  df <- read.csv("https://github.com/yhat/demo-lead-scoring/blob/master/data/leads-and-customers.csv?raw=true")
  head(df)

  df$is_manager <- grepl("manager|director|supervisor", df$job_title, ignore.case = TRUE)

  head(df)

  company_size_levels <- c("1-10", "11-50", "51-100", "101-250", "251-1000", "1000-10000", "10001+")
  df$company_size <- factor(df$company_size, levels=company_size_levels)

  for (f in c('is_manager', 'days_since_signup', 'visited_pricing', 'registered_for_webinar', 'attended_webinar', 'completed_form')) {
    print(f)
    print(table(df[,f], df$converted))
    print(paste0(rep("*", 80), collapse=""))
  }

  features <- c(
    "is_manager",
    "days_since_signup",
    "completed_form",
    "visited_pricing",
    "registered_for_webinar",
    "attended_webinar",
    "acquisition_channel",
    "company_size",
    "industry",
    "converted"
  )

  library(randomForest)


  logit <- glm(converted ~ ., data=df[,features])
  print(summary(logit))
# rf <- randomForest(converted ~ ., data=df[,features])


  probs <- predict(logit, newdata=df, type="response")
  grades <- cut(probs, 5, labels=c("F","D","C","B","A"), ordered_result=TRUE)

  lead_quality <- table(grades)
  o <- order(names(lead_quality))
  print(lead_quality[o])
}

# library(yhatr)
source("~/workspace/github.com/yhat/yhatr/R/yhatR.R")
yhat.library("randomForest")

model.predict <- function() {
  main()
}

yhat.config <- c(
  username="",
  apikey="",
  env=""
)

yhat.batchDeploy("lead_scorer")

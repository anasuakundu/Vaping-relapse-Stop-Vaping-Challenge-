library(dplyr)
library(devtools)
library(forcats)
library(tableone)
library(naniar)
library(mice)
library(xgboost)
library(ggplot2)
library(caret)
library(Hmisc)
library(pROC)
library(ROCR)
library(gbm)
library(survival)
library(SHAPforxgboost)
library(data.table)
library(shapviz)
library(reshape2)
library(lubridate)
library(Boruta)
library(survAUC)
library(survXgboost)
library(survex)
library(patchwork)
library(kernelshap)
library(gmodels)
library(shapr)
library(randomForestSRC)
library(caret)
library(survival)
library(survminer)

setwd("C:/Anya/IMS PhD/Thesis/Stop vaping challenge/Data 290425")

#load data
df.base <- read.csv('SUAPStopVapingChalle_DATA_2025-04-29_1336.csv')
summary(df.base)
str(df.base)
#N=2985, p=90

table(df.base$consent_yes_complete)
#Consent complete n=1588.

#Select participants who gave consent
df.base<- df.base[df.base$consent_yes_complete==2,]
#n=1588, 90 variables

#Drop unnecessary variables
df.base <- df.base %>% select(-c(record_id:consent_timestamp))%>% select(-c(consent_bl:baseline_timestamp))
#n=1588, p=78.

#removing particiants without userid
df.base <- df.base[!is.na(df.base$userid),]
#n=1568, p=78

#Selecting data and variables for main analysis
df.base <- df.base %>% select(c(userid, ecusv_1___1: ecusv_1___14, ecu4_1, ecu8_1: cur_otob_1, 
                                osu16_ecig_1, kab3_1:country, gender, sexorient, race___1:race___12))
#n=1568, p=57

#arranging by userid
df.base <- df.base %>% arrange(userid)
#some userid had duplicates, removing duplicates
df.base<-df.base%>% distinct(userid, .keep_all = TRUE)
#N=1488, p=57


###Importing mood and cravings data
mood<- read.csv('moods_and_cravings.290425.csv')
df1<- read.csv('moods_and_cravings.290425.csv')
#n=2504, p=8

df1 <- df1 %>% arrange(user_id)

###getting dataset with only first mood and craving recording
first_mood_craving <- df1 %>%
  group_by(user_id, challenge_id) %>%
  filter(created_at == min(created_at)) %>%
  select(user_id, challenge_id, mood, craving)


first_mood_craving <- first_mood_craving %>% rename(first_mood= mood, first_craving=craving)
#n=1,149, p=4

###making mood_trend and craving_trend variables
calculate_mood_trend <- function(mood) {
  if (length(mood) < 2) return(3)
  
  trend <- lm(mood ~ seq_along(mood))$coefficients[2]
  sd_mood <- sd(mood)
  
  if (sd_mood < 1) {
    return(1)
  } else if (trend > 0.5) {
    return(2)
  } else if (trend < -0.5) {
    return(0)
  } else {
    return(3)
  }
}

mood_trend <- df1 %>%
  arrange(user_id, challenge_id, created_at) %>%  # Ensure correct order over time
  group_by(user_id, challenge_id) %>%
  summarise(mood_trend = calculate_mood_trend(mood), .groups = "drop")
#n=1149, p=3


#craving
calculate_craving_trend <- function(craving) {
  if (length(craving) < 2) return(3)
  
  trend <- lm(craving ~ seq_along(craving))$coefficients[2]
  sd_craving <- sd(craving)
  
  if (sd_craving < 1) {
    return(1)
  } else if (trend > 0.5) {
    return(2)
  } else if (trend < -0.5) {
    return(0)
  } else {
    return(3)
  }
}

craving_trend <- df1 %>%
  group_by(user_id, challenge_id, created_at) %>%
  group_by(user_id, challenge_id) %>%
  summarise(craving_trend = calculate_craving_trend(craving), .groups = "drop")
#n=1149, p=3

##Creating updated_at dates
last_date <- df1 %>%
  arrange(user_id, challenge_id, created_at) %>% 
  group_by(user_id, challenge_id) %>%
  summarise(last_date = max(updated_at, na.rm = TRUE), .groups = "drop") %>%
  ungroup()

#joining all
df1 <- first_mood_craving %>%
  left_join(mood_trend, by = c("user_id", "challenge_id"))
df1 <- df1 %>%
  left_join(craving_trend, by = c("user_id", "challenge_id"))
df1 <- df1 %>%
  left_join(last_date, by = c("user_id", "challenge_id"))

#rename user_id
df1 <- df1 %>% rename(userid = user_id)
#n=1149, p=7

###########importing challenge data
challenge<- read.csv('challenges.290425.csv')
df2 <- read.csv('challenges.290425.csv')
summary(df2)
str(df2)
#n=8394, p=7
#Recoding status or censoring status 
table(df2$status)
#status=20 is completed challenges (n=6998), and status=10 is ongoing challenges (n=1101)
#recoding status
1-> df2$status[which(df2$status== 20)]
0-> df2$status[which(df2$status== 10)]

df2 <- df2 %>% arrange(user_id)
df2 <- df2 %>% rename(challenge_id = id)
df2 <- df2 %>% rename(userid = user_id)

####Linking df1 with df2, only keeping those which have mood and cravings data
merged_df <- df1 %>%
  left_join(df2, by = c("userid", "challenge_id"))
merged_df <- merged_df %>% ungroup()
#n=1,149, p=12

###changing date formats
merged_df <- merged_df %>%
  mutate(
    created_at = as.POSIXct(created_at, format = "%Y-%m-%d %H:%M:%S"),
    stopped_at = as.POSIXct(stopped_at, format = "%Y-%m-%d %H:%M:%S"),
    last_date = as.POSIXct(last_date, format = "%Y-%m-%d %H:%M:%S")
  )

###For status=1, time to relapse is at stopped_at, but for status=0, the time to relapse will be the last mood entry
merged_df <- merged_df %>%
  mutate(Time.elapsed = ifelse(status == 1,
                               as.numeric(difftime(stopped_at, created_at, units = "mins")),
                               as.numeric(difftime(last_date, created_at, units = "mins"))))

#subsetting challenge time at least 5 min
merged_df<- merged_df[merged_df$Time.elapsed>=5,]
#converting challenge time to days
merged_df$Time.elapsed<-merged_df$Time.elapsed/1440
#N=962, p=13

#dropping variables
merged_df <- merged_df %>% select(-c(stopped_at, notify_at, created_at, updated_at, last_date))
#N=962, p=8
summary(merged_df$Time.elapsed)
#no NA values

###getting a dataset with only common userid between df1 and df.base
df1 <- merge(df.base, merged_df, by = "userid")
###N= 527, p=64

#data cleaning
str(df1)
df1$challenge_id<- NULL

#past 30-day e-cigarette use
table(df1$ecu4_1)
#removing non-users and never users
df1<- df1 %>%
  filter(!(ecu4_1 == 5 | ecu4_1 == 6))
#all are now daily and weekly users
df1$ecu4_1<- NULL
#N=488, p=62

#age
table(df1$age)
summary(df1$age)
df1<- df1[which(df1$age>=15 & df1$age<=35),]
#n=387,p=62

###getiing number of particpants
df_base<-df1%>% distinct(userid, .keep_all = TRUE)
#311 particpants

#identifying userid that are outliers
summary<- df1%>% group_by(userid) %>% summarise( N=n(),.groups="drop")
table(summary$N)
###maximum no. of challenges by 1 individual1 was 6, so distribution good.

#reasons for quitting
1 -> df1$ecusv_1___4[which(df1$ecusv_1___4==1 | df1$ecusv_1___5==1)]
1 -> df1$ecusv_1___7[which(df1$ecusv_1___7==1 | df1$ecusv_1___8==1)]
1 -> df1$ecusv_1___13[which(df1$ecusv_1___1==1 | df1$ecusv_1___6==1 | df1$ecusv_1___9==1 | df1$ecusv_1___10==1 | 
                              df1$ecusv_1___11==1 | df1$ecusv_1___12==1 | df1$ecusv_1___13==1)]
df1 <- df1 %>% select(-c(ecusv_1___1, ecusv_1___5, ecusv_1___6, ecusv_1___8:ecusv_1___12, ecusv_1___14))

columns_to_update <- select(df1, c(ecusv_1___2, ecusv_1___3, ecusv_1___4, ecusv_1___7, ecusv_1___13)) %>%
  names()
df1 <- df1 %>%
  mutate(across(all_of(columns_to_update), ~ as.factor(.)))

#time to first vape
table(df1$ecu11b_1)
0 -> df1$ecu11b_1[which(df1$ecu11b_1==1)]
1 -> df1$ecu11b_1[which(df1$ecu11b_1==2 | df1$ecu11b_1==3)]
2 -> df1$ecu11b_1[which(df1$ecu11b_1==4 | df1$ecu11b_1==5 | df1$ecu11b_1==6)]
df1$ecu11b_1<- as.factor(df1$ecu11b_1)

#Number of puff
table(df1$ecu12a_1)
0 -> df1$ecu12a_1[which(df1$ecu12a_1==1)]
1 -> df1$ecu12a_1[which(df1$ecu12a_1==2)]
2 -> df1$ecu12a_1[which(df1$ecu12a_1==3 | df1$ecu12a_1==4)]
df1$ecu12a_1<- as.factor(df1$ecu12a_1)

#self-perceived addiction
table(df1$ecu16_1)
#not addicted and unknown status values are 3 and 1 respectively, hence, rate is lower than 5%
0 -> df1$ecu16_1[which(df1$ecu16_1==2)]
NA -> df1$ecu16_1[which(df1$ecu16_1==3 | df1$ecu16_1==4 | is.na(df1$ecu16_1))]
df1$ecu16_1<- as.factor(df1$ecu16_1)

#baseline intention to quit 
table(df1$ecu17_1)
NA-> df1$ecu17_1[which(df1$ecu17_1==4)]
0-> df1$ecu17_1[which(df1$ecu17_1==2 | df1$ecu17_1==3)]
df1$ecu17_1<- as.factor(df1$ecu17_1)

#past year quit attempts
table(df1$ecusv_2)
NA-> df1$ecusv_2[which(df1$ecusv_2==3 | df1$ecusv_2==4)]
0 -> df1$ecusv_2[which(df1$ecusv_2==2)]
df1$ecusv_2<- as.factor(df1$ecusv_2)

#device type
table(df1$epp1_1)
##level 8 (unknown) value is 0 and level 7 value (others) is 9, the total will be less than 5%, hence, obliterating these levels.
0 -> df1$epp1_1[which(df1$epp1_1==1 | df1$epp1_1==2)]
1-> df1$epp1_1[which(df1$epp1_1==3 | df1$epp1_1==4)]
2-> df1$epp1_1[which(df1$epp1_1==5 | df1$epp1_1==6)]
NA-> df1$epp1_1[which(df1$epp1_1==7 | df1$epp1_1==8 | is.na(df1$epp1_1))]
df1$epp1_1<- as.factor(df1$epp1_1)

#flavour
table (df1$epp7_1)
0-> df1$epp7_1[which(df1$epp7_1==6 | df1$epp7_1==7)]
1 -> df1$epp7_1[which(df1$epp7_1==1 | df1$epp7_1==2 | df1$epp7_1==3)]
2-> df1$epp7_1[which(df1$epp7_1==4 | df1$epp7_1==5 | df1$epp7_1==8 | df1$epp7_1==9)]
NA -> df1$epp7_1[which(df1$epp7_1==10 | is.na(df1$epp7_1))]
df1$epp7_1<- as.factor(df1$epp7_1)

#nicotine strength
table(df1$epp10_1)
0 -> df1$epp10_1[which(df1$epp10_1==1)]
1 -> df1$epp10_1[which(df1$epp10_1==2 | df1$epp10_1==3 | df1$epp10_1==4 | df1$epp10_1==5)]
2 -> df1$epp10_1[which(df1$epp10_1==6 | df1$epp10_1==7 | df1$epp10_1==8)]
3 -> df1$epp10_1[which(df1$epp10_1==9 | is.na(df1$epp10_1))]
#level 0 has only one value, so removing it
1 -> df1$epp10_1[which(df1$epp10_1==0 | df1$epp10_1==1)]
df1$epp10_1<- as.factor(df1$epp10_1)

#intensity of use
#Monthly vaping expense
summary(df1$epp13_1)
table(df1$epp13_1)
#finding outliers
Q1 <- quantile(df1$epp13_1, 0.25, na.rm= T)
Q3 <- quantile(df1$epp13_1, 0.75, na.rm= T)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
outliers <- df1$epp13_1[df1$epp13_1 < lower_bound | df1$epp13_1 > upper_bound]
summary(outliers)
##mimium value of outleirs is 170
boxplot(df1$epp13_1, main = "Boxplot of epp13_1", horizontal = TRUE)

#removing outliers more than 169
NA -> df1$epp13_1[which(df1$epp13_1>169)]

#Average e-liquid per week
summary(df1$epp14_1)
table(df1$epp14_1)
#finding outliers
Q1 <- quantile(df1$epp14_1, 0.25, na.rm= T)
Q3 <- quantile(df1$epp14_1, 0.75, na.rm= T)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
outliers <- df1$epp14_1[df1$epp14_1 < lower_bound | df1$epp14_1 > upper_bound]
summary(outliers)
##mimium value of outleirs is 70
boxplot(df1$epp14_1, main = "Boxplot of epp14_1", horizontal = TRUE)

#Removing outliers and entry less than 1 and more than 69
NA -> df1$epp14_1[which(df1$epp14_1<1 | df1$epp14_1>69)]

#Single pod lasting
summary(df1$epp15_1)
table(df1$epp15_1)
#finding outliers
Q1 <- quantile(df1$epp15_1, 0.25, na.rm= T)
Q3 <- quantile(df1$epp15_1, 0.75, na.rm= T)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
outliers <- df1$epp15_1[df1$epp15_1 < lower_bound | df1$epp15_1 > upper_bound]
summary(outliers)
##mimium value of outleirs is 20
boxplot(df1$epp15_1, main = "Boxplot of epp15_1", horizontal = TRUE)

#removing outliers more than 19 days
NA -> df1$epp15_1[which(df1$epp15_1>19)]

#current other substance use
columns_to_update <- select(df1, c(cur_csmk_1, cur_can_1, cur_alc_1, cur_wp_1, cur_otob_1)) %>%
  names()
for (col in columns_to_update) {
  0 -> df1[, col][df1[, col]==2 |df1[, col]==3 | df1[, col]==4]
}

#merging waterpipe with OTPs
table(df1$cur_otob_1)
1 -> df1$cur_otob_1[which(df1$cur_otob_1==1 | df1$cur_wp_1==1)]

df1 <- df1 %>%
  mutate(across(all_of(columns_to_update), ~ as.factor(.)))
df1$cur_wp_1<- NULL

#peer use
table(df1$osu16_ecig_1)
summary(df1$osu16_ecig_1)
0 -> df1$osu16_ecig_1[which(df1$osu16_ecig_1==1)]
1 -> df1$osu16_ecig_1[which(df1$osu16_ecig_1==2| df1$osu16_ecig_1==3)]
NA -> df1$osu16_ecig_1[which(df1$osu16_ecig_1==4 | is.na(df1$osu16_ecig_1))]
df1$osu16_ecig_1<- as.factor(df1$osu16_ecig_1)

#harm perception
summary(df1$kab3_1)
table(df1$kab3_1)
0 -> df1$kab3_1[which(df1$kab3_1==1|df1$kab3_1==2)]
1 -> df1$kab3_1[which(df1$kab3_1==3|df1$kab3_1==4)]
2 -> df1$kab3_1[which(df1$kab3_1==5 | is.na(df1$kab3_1))]
df1$kab3_1<- as.factor(df1$kab3_1)

#general and mental health, mood
summary(df1$ghealth_1)
summary(df1$mhealth1_1)
summary(df1$svc_mood)
columns_to_update <- select(df1, c(ghealth_1, mhealth1_1, svc_mood)) %>%
  names()
for (col in columns_to_update) {
  0 -> df1[, col][df1[, col]==4 |df1[, col]==5]
  1-> df1[, col][df1[, col]==1 |df1[, col]==2 | df1[, col]==3]
}

df1 <- df1 %>%
  mutate(across(all_of(columns_to_update), ~ as.factor(.)))

#Svc_cravings
table(df1$svc_cravings)
0 -> df1$svc_cravings[which(df1$svc_cravings==2|df1$svc_cravings==3 | df1$svc_cravings==4)]
###only 5 observations who relapsed had reported no cravings, so making them NA.
NA -> df1$svc_cravings[which(df1$svc_cravings==5 | df1$svc_cravings==6 | is.na(df1$svc_cravings))]
df1$svc_cravings<- as.factor(df1$svc_cravings)

#stress
table(df1$pstress_1)
0 -> df1$pstress_1[which(df1$pstress_1==1|df1$pstress_1==2)]
1 -> df1$pstress_1[which(df1$pstress_1==3|df1$pstress_1==4 | df1$pstress_1==5)]
df1$pstress_1<- as.factor(df1$pstress_1)

#gender
table(df1$gender)
df1$gender <- trimws(df1$gender)
0-> df1$gender[which(df1$gender=="he him"|df1$gender=="Helicopter man, male"| df1$gender== "male" | df1$gender=="Male" | df1$gender== "man"| 
                       df1$gender=="Man"|df1$gender=="MAN" | df1$gender=="men" | df1$gender=="Men" | df1$gender=="Boy")]
1-> df1$gender[which(df1$gender=="Cis woman" | df1$gender=="female" |df1$gender=="Female" | df1$gender=="Girl" | 
                       df1$gender=="woman" | df1$gender=="Woman" | df1$gender=="Woman She/Her" | df1$gender=="woman, female, girl"
                     | df1$gender=="women" | df1$gender=="Women"| df1$gender=="Her She" | df1$gender=="She/her" | df1$gender=="Woma")]

#only 3 transgender, combining with gender diverse and undisclosed status, to avoid less than 5%
2-> df1$gender[which(df1$gender=='Gender fluid'| df1$gender=='genderqueer' | df1$gender=='Non-binary' | df1$gender=='Non binary femme' |
                       df1$gender=='Trans' |  df1$gender=='Woman, two spirit' | df1$gender=="Man, woman" | 
                       df1$gender=='gender queer' |df1$gender=="bigender, gender non-conforming" | df1$gender=="Gigachad" | df1$gender=="Woman, hoe"| df1$gender=="" | df1$gender=="Wtf" | df1$gender=="Every one"|
                       df1$gender=="N/a"| df1$gender=="?" | df1$gender=="Emerson sister fucker"| is.na(df1$gender))]
df1$gender <- as.factor(df1$gender)

#sexual orientation
table(df1$sexorient)
df1$sexorient <- trimws(df1$sexorient)
df1$sexorient[df1$sexorient %in% c(
  "Helosexual, straight", "heterosexual", "sraight", "Staight", "straigh", "straight", "Straight",
  "straightish", "Striaght", "Hetero", "Heterosexual", "STRAIGHT", "Stright", "Hetrosexual", "Straights"
)] <- 0
df1$sexorient[df1$sexorient %in% c("bi", "Bi", "bisexual", "Bisexual", "Bisexal", "gay", "Gay", "gay/queer",
  "lesbian", "Lesbian", "polysexual", "pansexual", "Pansexual", "Pan","Queer", "not sure", "?"
)] <- 1
df1$sexorient[df1$sexorient %in% c( "", "Man", "Male", "N/a", "Non", "Wtf","Audreysexual", "Normal") | is.na(df1$sexorient)] <- 2

df1$sexorient <- as.factor(df1$sexorient)

#race
df1$race<-NA
0-> df1$race[which(df1$race___1==1| df1$race___2==1 | df1$race___3==1 | df1$race___4==1 | df1$race___5==1
                   | df1$race___6==1 | df1$race___7==1 | df1$race___8==1 | df1$race___9==1 | df1$race___10==1
                   | df1$race___12==1)]
1-> df1$race[which(df1$race___11==1)]
df1 <- df1 %>% select(-c(race___1:race___12))
df1$race <- as.factor(df1$race)

#checking variables
str(df1)
df1$country <- as.factor(df1$country)
df1$status <- as.factor(df1$status)
df1$mood_trend<- as.factor(df1$mood_trend)
df1$craving_trend<- as.factor(df1$craving_trend)
##n=387, p=41

###average no. of quit attempts
summary<- df1%>% group_by(userid) %>% summarise(  N=n(),.groups="drop")
summary %>% summarise(MeanN=mean(N), sd= sd(N),.groups="drop")
#average no. of quit attempts 1.24 (sd 0.661)

###getiing number of particpants
df1_base<-df1%>% distinct(userid, .keep_all = TRUE)
#311 participants

###Descriptive statistics
var<- select(df1_base, c('age', 'country', 'gender', 'sexorient', 'race', 'ecu11b_1', 'ecu16_1', 'svc_cravings'))

ftable <- CreateTableOne(data=var,includeNA =F,
                         test=T)
print(ftable, showAllLevels = T)

##examining descriptive statistics for challenges
CreateTableOne(data=df1,includeNA =F, test=T)
##All levels are at least 5%

##Time statistics
summary(df1$Time.elapsed)
#minimum 0.004, max 194, average 4.04 days, sd 15.69, median 0.54

###descriptive statistics
ftable1 <- CreateTableOne(data=df1,includeNA =F,strata="status", test=T)
print(ftable1, showAllLevels = T)

###removing unnecessary variables
df1$userid<- NULL
##Drop country
df1$country<- NULL
#n=387, p=39

###saving unimputed data
write.csv(df1, "unimputed_dataset_final.csv")

###Missing data
sapply(df1, function(x) sum(is.na(x)))

vis_miss(df1,sort_miss=TRUE)
#3.1% missing data

q <- df1 %>% summarise_all(~sum(is.na(.)))
q2 <- t(q)
q3 <- data.frame('Number_of_missingness'= q2[,1],
                 'percent_missingness'=round(q2[,1]/nrow(df1)*100, digit=2))
# sort
q3 <- q3[order(q3$percent_missingness,q3$Number_of_missingness),]

# how many vars had >=5% missing?
m <- q3[q3$percent_missingness>=5,]
dim(m)
##5 variables had more than 5% missing data, ecu8_1, ecu14_1, race, epp15_1, epp13_1

####data imputation
init <- mice(df1, maxit = 0)
meth <- init$method
predM <- init$predictorMatrix

# Exclude outcome as predictor
predM[, c("Time.elapsed", "status")] <- 0

# methods for different variables
pmm <- c("ecu8_1", 'epp14_1', 'epp15_1', 'epp13_1')
polr <- c('ecu12a_1', 'epp1_1', 'epp7_1')
logreg<- c('race', 'ecu17_1', 'ecusv_2', 'ghealth_1','osu16_ecig_1','mhealth1_1', 'pstress_1',
           'svc_mood', 'svc_cravings', 'ecu16_1')

meth[pmm] <- "pmm"
meth[polr] <- "polr"
meth[logreg] <- "logreg"

# Set seed and perform imputation
set.seed(123)
imputed <- mice(df1, method = meth, predictorMatrix = predM)
summary(imputed)
df1 <- complete(imputed,1)

sapply(df1, function(x) sum(is.na(x)))
vis_miss(df1,sort_miss=TRUE) 
##All imputed

#saving imputed df1aset
write.csv(df1, "imputed_dataset_final.csv")





# Company Carbon Emissions & Commitment to NetZero 

## Problem Statement

 
[Entelligent](https://www.entelligent.com) is a quantitative analytics firm that helps financial institutions measuring and managing climate transition risk. Transition risk refers to the ability to adapt to changing policies, technologies, market preferences and societal norms. There is considerable uncertainty regarding how these changes will look like to correcting course with respect to climate change. 

To address this issue, Entelligent designed a scenario analysis top-down and bottom-up technology.
The top-down model uses Integrated Assessment Models to understand how the changing energy mix under various climate scenarios could impact company share price returns. The bottom-up analysis includes the data on company’s scope 1, scope 2 and scope 3 emissions, providing a snapshot of a company’s current situation. 

Because historical company-level emissions information are not always available, serving as a climate policy manager and data scientist fellow at Entelligent, I wanted to find additional predictors that could help estimate company emissions.
I was interested to conclude whether country-level information (e.g. emission factor or emission per GDP) and company climate action target (e.g. if the company committed to NetZero) could both have a relation with company emissions. 

My assumption is that companies’ greenhouse gas emissions could depend from both external and internal factors. For instance, a company that operates in a country whose energy mix relies more on fossil fuels, could have greater emissions than a peer (e.g. same industry and/or market capitalization) that works in a country powered mainly by renewable energy sources. Moreover, a company that presents a lower level of emission relative to its peers, could be more inclined to commit to NetZero.


## Executive Summary

To take a look at these relationships, I developed first a Ttest analysis on all additional predictors; then I developed a multiple linear regression model that was using only the predictors that passed Ttest. Lastly, to further investigate the relationship between company carbon emissions and NetZero commitment, I developed a classification model to predict whether or not a company committed to NetZero based on their emissions and few other predictors. The classification model has been deployed over a streamlit app with the idea of developing a user-friendly tool that could help a non-technical audience understanding which key company characteristics could increase the likelihood of NetZero commitments. 

The following notebooks require the use of Pandas, Matplotlib, Seaborn and Scikit-learn.

#### 01_Data Import and Cleaning_CountryEmissions

__Data Sources__

There were many potential options for sourcing country level data;the best choice being [Our World In Data](https://ourworldindata.org/about). The thoroughness, transparency, and accessibility made it a clear choice. From this source I gathered the following country level dataset: i) **Emissions Factors**: amount of carbon dioxide equivalent emitted per unit of energy produced (kg/kwh); ii) **Emissions per GDP**: amount of carbon dioxide equivalent emitted per unit of GDP produced  (kg per PPP); this metric suggest how carbon-intensive a country's economy is (often called as the carbon intensity of economies).GDP measured in cnstant 2011 international-dollars. iii) **Emissions by Sector**: tons of CO2e per sector. iv) **Emissions by Fuel**: tons of CO2 emitted per fuel type associated with energy and industrial production. iv) **Total GHG Emissions**; tons of CO2e  per country, including emissions form land use change. GHG includes carbon dioxide, methane, nitrouse oxide and F-gases. vi) **Consumption-based CO2 emissions**: consumption based are adjusted for trade;the difference between a country’s consumption-based emissions and production-based emissions. This means it is the net trade of emissions.It is equal to consumption-based emissions minus production-based emissions in any given year. This means net importers of emissions have positive values. Net exporters have negative values. 

__Cleaning and Merging__

Entelligent data was provided on a quarterly basis. By contrast, country level information was provided yearly. Thus, I needed to upsample all countries dataset down to quarter and merged everything in one uniqe dataframe. Further cleaning included correcting quarter date, renaming columns, correcting data types and filtering by countries and timestamp (2012-2020)
Before merging this dataset with company-level information I also did the following:
1. Filled in  roughly ~10%-20% of null value for some of country variables through IterativeImputer. 
2. Added country [ISIN code](https://www.isin.net/country-codes/) obtained through webscraping.


#### 01_Data Import  Cleaning and Feature Engineering_
  
__Data Sources__

Company Level information such as industry, region, market capitalization and carbon emissions were provided by Entelligent on a quarterly basis for the period 2012-2020 for ~ 15,000 companies, across multiple benchmark, for a total of 1+ milion observations. Emission data was provided as three separate features: scope 1, scope 2 and scope 3 emissions. Scope 1 emissions refers to the greenhouse gases that a company emits directly, as produced from sources that are controlled or owned by the company itself (e.g. vehicle, boilers, furnaces). Companies also produce emissions indirectly by for example purchasing the electricity or energy for heating and cooling its own buildings. The energy carriers produced for company operations generates Scope 2 emissions. Scope 3 emissions instead, are the ones generated in a company value chain (suppliers and customers)

As per company climate action target I have used data coming from [The Science Based Targets Inititative](https://sciencebasedtargets.org/about-us), a partnership between CDP, the United Nations Global Compact, World Resources Institute (WRI) and the World Wide Fund for Nature (WWF). This initiative show companies and financial institutions how much and how quickly they need to reduce their greenhouse gas emissions 'to prevent the worst effect of climate change'. It is the largest and most important organization that gathers climate commitments submitted by ~ 3000 companies.The dataset included the following: i)if the company committed to **netzero**; ii) if the company set an **emission reduction target** that is in line with either/or a 2C, well-below 2C and 1.5C climate scenario; iii) **target year(s)**: the roadmap to achieve its climate commitment. 

__Cleaning and Merging__

Company emissions and company industry information were provided in two separate datataset. 
Because of the limitation of company climate commitment data, and interested to take a look at companies spread across different regions, I decided to first filtering companies that were belonging only to the MSCI ACWI benchmark (~2,500 companies for a total of ~46,000 observations). And then merge this dataset with company carbon emissions, country emissions and company climate commitment. Out of ~2500 companies I could only find climate actions commitment for roughly ~1000 firms. For the remaining ~2000 companies  I assumed that they did not make any climate commitments yet. 
Further cleaning included correcting quarter date, renaming columns, correcting data types.


__Feature Engineering__

As per feature engineering I did the following:

1.Scope 1, Scope 2 and Scope 3 emissions were provided as three separate targets. I also decided to create two additional targets by i) summing scope 1 with scope 2 emissions and ii) summing scope 1 together with scope 2 and scope 3. The goal was to potentially improve model interpretability and perfomances.  

2. Transformed company market capitalization from continuous to categorical, with the following threshold: i) large cap if above 10B$; ii) medium cap if between 2B$-10B$; iii) small cap if below 2B$

3. companies that submitted a climate commitment to the SBTI, needed to indicate the following: i) temperature target (1.5 degrees, well-below 2 degrees or 2 degrees); the roadmap to reduce carbon emissions to achieve temperature target (year(s)); if they already committed ot netzero emission. Because companies could indicate one or more years in their roadmap, I decided to creates the following group of roadmaps:
a. for companies that indicated only one year: i) 2030; ii) later than 2030; iii) earlier than 2030
b. for companies that indicated more than one year:i) 2030 and earlier; ii) 2030 and later; iii) earlier and later than 2030
Lastly I added this column to the temperature target's one. Although this classification improved the classification model accuracy, to balance out model interpretability and accuracy I decided to substitute these predictors with the following group classification:  i)not_taking_action for companies that SBTI did not have information for; ii) targets_set: for companies whose emission reduction roadmap is approved by the SBTI; iii) committed: for companies whose emission reduction roadmap is still pending. 

4.Created a new region feature by following Entelligent classification

5.Created a binary country economy feature by following [MSCI classification] (source: https://www.msci.com/our-solutions/indexes/market-classification).Countries were then divided as having either an emerging or developed economy

6. Although not included in this jupyter notebook, I also needed to dummify few predictors: i) for the **Regression Model** gics_sector_name, gics_sub_industry_name, region, company_cap, country_economy, Entity, net_zero_commited, target_status_class_year. This dummification can be found in regression model notebook. ii) For the **Classification Model** gics_sector_name, region, company_cap, country_economy, target_status_class_year. This dummification can be found in classification model notebook.


#### 03_EDA&TtestAnalysis

In the process of EDA, I looked at both correlation and t-test to determine which variables I wanted to include in the regression model. However, because I was interested to detect potential relationship between country level emission data and company climate commitments with company emissions, I decided to select only predictors that passed the ttest. 
The T-test analysis also helped me understand the final shape of my targets. As previously mentioned, Scope 1, Scope 2 and Scope 3 emissions were provided as three separate targets. After running several T-tests, the adjusted R2s were suggesting that predictions were more accurate by combining  Scope 1 and Scope 2 emissions together while leaving as a separate additional target Scope 3 emissions. The decision of leaving Scope 3 as a separate target, was also supported for the following reasons: i) while scope 1 and scope 2 emissions are produced for company operation, scope 3 emissions are generated through company wider value chain. ii) according to [ghg protocol corporate standard](https://ghgprotocol.org), scope 1 and two are mandatory to report, whereas scope 3 is voluntary. iii) Scope 3 is the hardest to measure and monitor, and thus less accurate. 
This notebook is structured in three different sections:i) section 1 where I looked at correlation factors; 2) section 2 where I conduct the T-test analysis and 3) section 3 where I take a closer look at the dataset.

#### 04_EDA for Classification

In this Jupyter Notebook I am taking a closer look at relationship between targets and predictors of the classification model. There are  two separate sections: 1) a general section where I am importing python libraries and creating a series of functions that I use throughout the notebook for data visualization. 2) The eda section is where I analyze the distribution of companies that commit and did not commit to netzero by region, sector, market capitalization, country economy and emission level. 


#### 05_Modeling_LinearRegression

This notebook shows the relationship between both country and company level information with company scope 1&2 and scope 3 emissions. Features selection through ttest had undoubtely introduced bias to the linear regression model. However, the main scope of this analysis was to detect a pattern between new potential predictors and targets. From here, the decision of choosing a more biased model to favorite model interpretability. The model ultimately selected was a multiple Linear regression model. 
The notbook presents four different sections: i) In the first section I import library and prepare the dataset for linear regression analysis; ii) in section 2 is where I assess the relationship between the additional predictors with companies' scope 1 and 2 emissions; iii)in section 3 is where I assess the relationship between the additional predictors with companies' scope 3 emissions; iv) the last section presents the conclusion of the analysis.

#### 06_Modeling_Classification

In this notebook I develop a classification model to predict whether or not a company commited to netzero. 
To do that, I  needed to find the right balance between model accuracy and model interpretability for a non-technical audience. Therefore, I did not put attention at correlation or ttest factors. Rather, at how intutitive the predictors were, while still guaranteeing high accuracy in the classification model. To select the model that could offer best accuracy and recall scores, I assessed the following classification model: Logistic Regression, DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier and KNeighborsClassifier. The model ultimately selected for the streamlit app was the RandomForestClassifier. The notebook is structured in three different sections:i) In the first section I import library and prepare the dataset for classification analysis; ii) in section 2 I am creating a pipeline to evaluate which model perform better without gridsearching. Models with better perfomance are passed in section 3; iii) in this section I am gridsearching through the models selected in Section 2. The model that presents best accuracy and recall scores is used for the streamlit application

#### _Webscraping

In this jupyter notebook there is the code for webscraping country ISIN code that I needed to merge country-level information with company-level information


## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**t_date**|*object*|msci_company_country.csv|timestamp on a quarterly basis|
|**fsym_id**|*object*|msci_company_country.csv|unique company id|
|**gics_sector_name**|*object*|msci_company_country.csv|Sector as described by the Global Industry Classification Standard (GICS)|
|**gics_sub_industry_name**|*object*|msci_company_country.csv|Industries as described by the Global Industry Classification Standard (GICS|
|**returns**|*float*|msci_company_country.csv|quartely company returns (%)|
|**region**|*object*|msci_company_country.csv|classification of geographic region as described by MSCI|
|**company_cap**|*object*|msci_company_country.csv|company market capitalization|
|**country_economy**|*object*|msci_company_country.csv|country type of economy as described by MSCI classification|
|**scope_1_tonnes**|*float*|msci_company_country.csv|Company scope 1 emissions (Thousand of Tonnes)|
|**scope_2_tonnes**|*float*|msci_company_country.csv|Company scope 2 emissions (Thousand of Tonnes)|
|**scope_3_tonnes**|*float*|msci_company_country.csv|Company scope 3 emissions (Thousand of Tonnes)|
|**scope1_2_emission**|*float*|msci_company_country.csv|company scope 1 and 2 emissions (Thousand of Tonnes)|
|**all_emission**|*float*|msci_company_country.csv|company scope 1,2 and 3 emissions (Thousand of Tonnes)|
|**net_zero_committed**|*object*|msci_company_country.csv|whether or not a company committed to netzero emission|
|**target_status_class_year**|*object*|msci_company_country.csv|this indicates whether a company emission reduction roadmap has been approved, still pending or has not been submitted|
|**emissions_lucf**|*float*|msci_company_country.csv|tons of CO2e  per country, including emissions form land use change. GHG includes carbon dioxide, methane, nitrouse oxide and F-gases.|
|**emissions_factor(kg/kwh)**|*float*|msci_company_country.csv|amount of carbon dioxide equivalent emitted per unit of energy produced (kg/kwh) per country|
|**emissions_gdp(kg/$ppp)**|*float*|msci_company_country.csv|amount of carbon dioxide equivalent emitted per unit of GDP produced  (kg per PPP) per country|
|**emissions_in_trade**|*float*|msci_company_country.csv|consumption-based CO2 emissions adjusted for trade per country|
|**co2_oil**|*float*|msci_company_country.csv|tons of CO2 emitted per oil associated with energy and industrial production per country|
|**co2_cement**|*float*|msci_company_country.csv|tons of CO2 emitted per cement associated with energy and industrial production per country|
|**co2_coal**|*float*|msci_company_country.csv|tons of CO2 emitted per coal associated with energy and industrial production per country|
|**co2_gas**|*float*|msci_company_country.csv|tons of CO2 emitted per gas associated with energy and industrial production per country|
|**agriculture**|*float*|msci_company_country.csv|tons of CO2e per agriculture sector per country|
|**land_use_forestry**|*float*|msci_company_country.csv|tons of CO2e per land use and forestry per country|
|**waste**|*float*|msci_company_country.csv|tons of CO2e per waste per country|
|**industry**|*float*|msci_company_country.csv|tons of CO2e per industry per country|
|**manufact_construction**|*float*|msci_company_country.csv|tons of CO2e per manufacturing and construction per country|
|**transport**|*float*|msci_company_country.csv|tons of CO2e per transportation per country|
|**electr_heat**|*float*|msci_company_country.csv|tons of CO2e per production of electricity and heat per country|
|**buildings**|*float*|msci_company_country.csv|tons of CO2e per buildings per country|
|**fugitive_emission**|*float*|msci_company_country.csv|tons of CO2e per fugitive emissions from energy production per country|
|**other_fuel_combustion**|*float*|msci_company_country.csv|Energy-related emissions from the production of energy from other fuels expressed in tons of CO2e per country|
|**aviation_shipping**|*float*|msci_company_country.csv|tons of CO2e per aviation and shipping per country|
|**Entity**|*object*|msci_company_country.csv|countries|



## Conclusion  

__Regression Model__

As previously mentioned, features selection through ttest had undoubtely introduced bias to the linear regression model.
Infact, the R2 score for predicting scope 1 and scope 2 emission is ~.69 while R2 for scope 3 is ~.71. By contrast, if I would include all predictors and apply a log transformation to both targets I would  obtain an R2 of respectively .72 and .90. With a Random Forest model I would instead achieve a R2 score  of ~.94 for scope 1 and 2 and ~.95 for scope 3. However, the main scope of this analysis was to detect a pattern between new potential predictors and emission targets. Below there are two tables that summarize the relationship between predictors for the two separate targets, scope1&2 and scope 3 emissions. 
The coefficient value of each predictor shows how much company emissions (in thousand of tonnes) would increase (for positive values) or decrease (for negative values), if we would hold everything else constant. For instance, holding everything else constant, a company that did not commit to netzero tends to emit ~600ktonnes of scope 3 emissions more than a company that did commit to netzero. Viceversa, a company that did commit to netzero tends to emit  ~175kt  of scope 1, 2 emissions than companies that did not commit to Netzero. By looking at the distribution of scope 1&2 and scope 3 emissions at the aggregate level, we can see a similar pattern: the median of scope 1&2 emissions for companies that commit to netzero is greater than the median of scope 1&2 emissions for companies that did not commit to netzero. By looking at Scope 3 emissions we instead see the exact opposite pattern. This relationship seems to suggest that companies that have higher level of scope 3 emissions and lower level of scope 1,2 emissions are less likely to commit to netzero. The rational behind this conclusion is explained below, in the classification model section.  

|**Predictors For Scope1&2**|**Coefficient Value**
|country_economy_developed|202.285835
|net_zero_committed_Yes|174.960675
|emissions_in_trade|-15.986954
|net_zero_committed_No|-269.747985


|**Predictors For Scope3**|**Coefficient Value**
|net_zero_committed_No|613.450781
|country_economy_developed|183.479648
|emissions_in_trade|63.656397
|net_zero_committed_not_taking_action|-182.224726
|target_status_class_year_not_taking_action|-182.224726
|net_zero_committed_Yes|-431.226056

__Classification Model__

The goal was to develop an application that could show if a company commits to NetZero based on its emissions level and few and intuitive company characteristcs while also guaranteeing high accuracy score. The scores I was interested to maximize were both accuracy and recall. I needed a model that could make as many true predictions as possible while minimizing false negatives; I'd rather label a company as being committed to netzero when it is not than the opposite. To select the model that could offer best accuracy and recall scores, I did the following: i) created a pipeline with a series of classification model, and select the ones that were performing better without gridsearching. This pre-filtering made me remove the logistic regression and AdaBoost classifier from the original list of classification models. The difference in the accuracy score was pretty substantial: ~.60 for both against ~.85 for the others.  ii) gridsearched the remaining following models: DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, KNeighborsClassifier; and chosing the model that was maximing accuracy and recall score. Below there is a summary with all different scores. The model ultimately selected was the RandomForestClassifier. iii) after further model-tuning, the random forest model achieved .90 of accuracy and .89 of recall. 

|**Model**|**Accuracy Score**|**Recall Score**|
|Logistic Regression|.62| .41
|DecisionTreeClassifier|.86|.84
|BaggingClassifier|.87|.84
|RandomForestClassifier|.90|.89
|AdaBoostClassifier|.64|.44
|KNeighborsClassifier|.87|.86

The Random forest modeling is harder to interpret than a multiple linear regression; however, by looking at its most important features (the ones that are dropping the most the gini score), it is clear that there is a strong relationship between company emissions and companies commitment to NetZero. Confirming the conclusions of the regression models. 
Through the streamlit app, it is interesting to notice how the probability of a company to commit to netzero changes by increasing or decreasing the scope 1 2 and 3 emissions sliders. Infact, by increasing the scope 3 emissions and decreasing the scope 1 and 2 emissions it is more likely that company won't commit to netzero. By contrast, if you reduce scope 3 emissions and increase scope 1 and 2 emissions, it is more likely that a company does commit to netzero. As a reminder scope 3 emissions are the ones that relate to company supply chain; i.e. the emissions that a company has less control on. These findings suggest
that the likelihood of a company to commit to netzero could depend on the level of control a company can have on its own carbon footprint. For instance, for companies with a lower level of scope 3 emissions but a greater level of scope 1 and 2 emissions could be easier to commit to netzero as they would have more control on their own carbon footprint. 



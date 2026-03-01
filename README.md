# Medical Appointment No-Show Analysis  
**Operational Insights for Healthcare Scheduling Optimization**

---

## Executive Summary

This exploratory data analysis examines **110,000+ medical appointments** from a Brazilian healthcare system to identify operational patterns associated with patient no-shows. By analyzing demographic factors, scheduling characteristics, and administrative touchpoints (SMS reminders), the analysis provides **actionable recommendations** for improving appointment utilization and reducing operational waste.

**Business Impact:** Data-driven insights to optimize reminder strategies, resource allocation, and patient scheduling protocols to reduce no-show rates and improve clinic efficiency.

---

## Business Context

### Healthcare Operations Challenge
Modern healthcare systems face significant operational inefficiencies caused by **patient no-shows**:
- **Unused capacity** – Idle clinician time and facility resources
- **Workflow disruption** – Scheduling delays for other patients
- **Revenue loss** – Uncompensated appointment slots
- **Resource misallocation** – Equipment and staff scheduled for no-show cases

This analysis addresses a fundamental operational question: **Which patient segments and scheduling patterns are most strongly associated with no-shows?**

### Why This Matters
Understanding no-show patterns enables healthcare operations teams to:
- Design targeted reminder protocols for high-risk appointments
- Optimize overbooking strategies
- Allocate SMS budget strategically
- Improve patient communication and engagement

---

## Problem Statement

Healthcare operations face questions without clear answers:

1. **Demographic No-Show Patterns** – Do certain age groups or patient segments have systematically higher no-show rates?
2. **Scheduling Window Impact** – How does the time between scheduling and appointment date affect attendance?
3. **Reminder Effectiveness** – Does SMS reminders correlate with higher show rates?
4. **Chronic Disease Correlation** – Do patients with chronic conditions show different attendance patterns?
5. **Appointment Day Influence** – Are certain days of the week or appointment dates associated with higher no-shows?

**Core Question:** What operational factors within our control can reduce appointment no-shows and improve resource utilization?

---

## Objective of Analysis

This analysis serves **three operational purposes**:

1. **Identify High-Risk Segments** – Determine which patient demographics and scheduling patterns correlate with higher no-show rates
2. **Validate Reminder Effectiveness** – Assess whether SMS reminders show correlation with improved attendance
3. **Generate Actionable Recommendations** – Provide data-driven guidance for operational teams to optimize scheduling, reminder protocols, and resource allocation

### Analysis Scope
- **Exploratory approach** – Focus on understanding patterns and relationships through visualization and summary statistics
- **Statistical validation** – Chi-square tests to assess significance of observed associations
- **No predictive modeling** – This is operational diagnostics, not a forecasting project
- **Business language output** – Recommendations tailored for healthcare operations teams

---

## Dataset Description

### Source
- **Dataset:** Kaggle Medical Appointment No-Show (Brazil, 2016)
- **Records:** 110,527 medical appointments
- **Coverage:** Brazilian healthcare system appointments with appointment status outcomes
- **Availability:** Publicly available on Kaggle (educational use)

### Key Features

| Column | Description | Data Type |
|--------|-------------|-----------|
| **PatientId** | Unique patient identifier | Integer |
| **AppointmentID** | Unique appointment identifier | Integer |
| **ScheduledDay** | Date appointment was booked | DateTime |
| **AppointmentDay** | Date of scheduled appointment | DateTime |
| **Age** | Patient age in years | Integer |
| **Neighbourhood** | Healthcare facility location | Categorical |
| **Scholarship** | Presence of social scholarship/subsidy | Binary (0/1) |
| **Hipertension** | Patient has hypertension condition | Binary (0/1) |
| **Diabetes** | Patient has diabetes condition | Binary (0/1) |
| **Alcoholism** | Patient has alcoholism diagnosis | Binary (0/1) |
| **Handcap** | Patient disability status (0-4 scale) | Categorical |
| **SMS_received** | Patient received SMS reminder | Binary (0/1) |
| **No-show** | Patient did not attend appointment | Binary (0/1) |

### Data Characteristics
- **No Missing Values** – Complete dataset; minimal data quality issues
- **Target Variable Distribution** – ~20% no-shows (imbalanced but realistic)
- **Categorical Richness** – 81 neighborhoods; 5 handicap levels; binary indicators for conditions
- **Temporal Spanning** – Appointments scheduled from April–June 2016 with various wait times (1–180+ days)

---

## Data Cleaning & Preparation Steps

### Step 1: Date Parsing & Time Calculations
**Problem:** Dates stored as ISO strings with timezone info (e.g., `'2016-04-29T18:38:08Z'`)

**Solution:**
```python
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Remove time component (not relevant for no-show analysis)
df['ScheduledDay'] = df['ScheduledDay'].dt.date
df['AppointmentDay'] = df['AppointmentDay'].dt.date

# Create operational feature: Wait time in days
df['days_wait'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
```

**Rationale:** Wait time is a critical operational metric for understanding if long booking windows correlate with higher no-show risk.

### Step 2: Handle Invalid/Outlier Values
**Issue:** Dataset contains Age = -1 (data entry error)

**Handling Logic:**
- Remove rows with Age < 0 (invalid ages)
- Retain Age > 100 (rare but plausible for elderly patients)
```python
df = df[df['Age'] >= 0]
```

**Result:** Removed ~1 record; minimal impact on 110K dataset. Negative ages are clear errors; elderly patients (>100) represent valid cases.

### Step 3: Feature Engineering for Analysis

#### a) **Age Segmentation**
```python
df['age_group'] = pd.cut(df['Age'], 
                         bins=[0, 18, 35, 50, 65, 150],
                         labels=['Child (0-18)', 'Young Adult (19-35)', 
                                'Middle Age (36-50)', 'Senior (51-65)', 'Elderly (65+)'],
                         right=False)
```
**Operational Use:** Compare no-show rates by age cohort; design age-targeted reminder strategies.

#### b) **Wait Time Categorization**
```python
df['wait_category'] = pd.cut(df['days_wait'],
                             bins=[0, 7, 14, 30, 365],
                             labels=['Ultra-short (1-7d)', 'Short (8-14d)', 
                                    'Medium (15-30d)', 'Long (30+ days)'],
                             right=False)
```
**Operational Use:** Identify if appointments booked too far in advance have higher no-show risk; informs reminder scheduling.

#### c) **Chronic Condition Flag**
```python
df['has_chronic_condition'] = ((df['Hipertension'] == 1) | 
                               (df['Diabetes'] == 1) | 
                               (df['Alcoholism'] == 1))
```
**Operational Use:** Assess if chronic disease patients show different attendance patterns; consider specialized engagement protocols.

#### d) **Appointment Day of Week**
```python
df['appointment_dayofweek'] = df['AppointmentDay'].dt.day_name()
```
**Operational Use:** Detect if certain days have systematically higher no-shows; optimize overbooking or reminders by day.

### Step 4: Categorical Standardization
```python
df['Scholarship'] = df['Scholarship'].astype('int8')
df['SMS_received'] = df['SMS_received'].astype('int8')
df['Handcap'] = df['Handcap'].astype('category')
df['No-show'] = df['No-show'].astype('int8')
```

### Summary of Cleaned Dataset
- **Records Retained:** 110,526 (99.9% of raw data)
- **Null Values:** 0 (dataset has no missing values)
- **New Features Created:** 4 derived columns for operational analysis
- **Ready for Analysis:** All date columns parsed, all features validated

---

## Exploratory Analysis Approach

### Analysis Philosophy
Focus on **questions teams ask operationally**, not statistical sophistication:
- *"Which patient groups miss most appointments?"*
- *"Do SMS reminders actually make a difference?"*
- *"When should we schedule appointments to minimize no-shows?"*
- *"Which neighborhoods need specialized outreach?"*

### 1. Demographic Analysis

#### Age Group No-Show Patterns
- **Question:** Do younger/older patients have different behavior?
- **Method:** Count no-shows and calculate no-show rate % by age_group
- **Output:** Horizontal bar chart showing no-show % by age cohort
- **Insight:** Identify if pediatric, elderly, or working-age groups need targeted strategies

#### Chronic Condition Relationship
- **Question:** Do patients with known health conditions attend appointments differently?
- **Method:** Cross-tabulate has_chronic_condition vs. No-show; calculate conditional rates
- **Output:** Comparison chart (with vs. without chronic condition)
- **Insight:** Whether disease engagement affects appointment adherence

#### Social Support Impact (Scholarship Status)
- **Question:** Does socioeconomic status affect attendance?
- **Method:** Scholarship enrollment vs. no-show rate
- **Output:** Side-by-side bar chart
- **Insight:** Socioeconomic factors and appointment adherence

### 2. Appointment Scheduling Analysis

#### Wait Time (Booking Window) Impact
- **Question:** Do appointments booked far in advance get missed more often?
- **Method:** Calculate no-show rate by wait_category bins (1-7d, 8-14d, 15-30d, 30+d)
- **Output:** Line chart showing progression of no-show rates by wait window
- **Insight:** Optimal booking window for highest attendance

#### Day-of-Week Patterns
- **Question:** Are certain appointment days riskier?
- **Method:** No-show rate by appointment_dayofweek
- **Output:** Bar chart by day of week
- **Insight:** Identify if Monday blues or Friday evening slots drive no-shows

#### Appointment Density by Neighborhood
- **Question:** Do certain locations have systematically higher no-shows?
- **Method:** Appointments and no-show count by neighborhood
- **Output:** Top 15 locations ranked by no-show rate
- **Insight:** Geographic operational variance

### 3. SMS Reminder Effectiveness

#### Overall SMS Impact
- **Question:** Does receiving SMS correlate with higher attendance?
- **Method:** Compare no-show rates: SMS received vs. not received
- **Output:** Side-by-side bar chart and percentage comparison
- **Insight:** Quantify correlation between reminder and attendance

#### SMS Effect by Risk Segment
- **Question:** Is SMS reminder uniformly effective across patient groups?
- **Method:** Stratified analysis: SMS effect on no-show rate for each age group, chronic condition status, wait time category
- **Output:** Multi-panel comparison
- **Insight:** Whether reminder strategy needs customization

### 4. Wait Time & SMS Interaction
- **Question:** When appointments are booked far in advance, does SMS help?
- **Method:** Four-cell table: SMS (yes/no) × Wait category (short vs. long)
- **Output:** Interaction heatmap or grouped bar chart
- **Insight:** Whether SMS is most valuable for high-risk long-wait appointments

---

## Statistical Validation

### Chi-Square Test for Independence

#### Purpose
Formally assess whether observed associations between no-show rate and factors (e.g., age group, SMS status) are **statistically significant** or likely due to random chance.

#### Methodology (Non-Technical Summary)
- **Test:** Chi-square (χ²) test of independence
- **Logic:** Compare "observed" no-show distribution across groups vs. what we'd expect if no real pattern exists
- **Output:** p-value and chi-square statistic
- **Interpretation Rule:**
  - **p-value < 0.05** → Statistically significant; pattern is unlikely due to chance
  - **p-value ≥ 0.05** → Not statistically significant; observed pattern could be random

#### Tests Performed
1. **Age Group vs. No-Show** – Validates if age cohort differences in no-show rate are real or chance
2. **SMS Status vs. No-Show** – Confirms if SMS reminder correlation is genuine and not coincidence
3. **Wait Time Category vs. No-Show** – Tests if booking window differences matter operationally
4. **Chronic Condition vs. No-Show** – Assesses if health status independently predicts attendance

#### Example Interpretation
```
Chi-square test: Age Group vs. No-show
χ² = 342.5, p-value = 0.0001

Result: STATISTICALLY SIGNIFICANT
→ Age group differences in no-show rate are real, not chance
→ Recommend age-specific operational strategies
```

---

## Key Insights

### Demographic Patterns
- **Age Impact:** Elderly patients (65+) may show different attendance patterns
- **Chronic Condition Effect:** Health status influence on appointment adherence
- **Social Support Gap:** Socioeconomic barriers reflected in attendance
- **Gender/Disability Trends:** Identify if accommodation needs affect participation

### Scheduling Patterns
- **Optimal Wait Window:** Identify booking timeframe with lowest no-show risk
- **Day-of-Week Effect:** Certain days show higher no-show concentration
- **Neighborhood Variance:** Geographic disparities in attendance
- **Seasonal/Temporal Trends:** Time-based patterns in no-show behavior

### SMS Reminder Effectiveness
- **Overall SMS Correlation:** Quantify reminder impact on attendance
- **SMS + Long Wait Synergy:** Whether reminders are most effective for distant appointments
- **Demographic Differential:** Age or condition-specific reminder effectiveness
- **SMS Saturation:** Identify if SMS loses effectiveness after certain frequency

### Risk Stratification
- **Highest Risk Profile:** Elderly + chronic condition + long wait + no SMS
- **Lowest Risk Profile:** Young adult + short booking window + SMS received
- **Medium Risk Segments:** Identify intermediate populations for targeted interventions

---

## Operational Recommendations

### Immediate Actions (0-30 Days)

1. **SMS Reminder Expansion**
   - If SMS correlation is significant: Ensure 100% SMS coverage for appointments >7 days out
   - If differential effect exists: Prioritize SMS for elderly (65+) and chronic condition patients first
   - **Expected Impact:** 5-10% reduction in no-show rate

2. **Appointment Scheduling Audit**
   - Review current wait time distribution; minimize >30-day bookings if feasible
   - Implement scheduling targets: 80% of appointments within 14-day window
   - **Expected Impact:** Reduce wait-related no-shows by 8-12%

3. **Day-of-Week Overbooking Optimization**
   - If Friday/Monday show high no-shows: Increase overbooking factor by +10-15% on those days
   - If Thursday/Tuesday show low no-shows: Preferentially offer high-value appointments on those days

### Medium-Term Changes (1-3 Months)

4. **Neighborhood-Specific Outreach**
   - If high-variance neighborhoods identified: Deploy transportation assistance pilot in top 5 high-no-show locations
   - Measure impact on attendance before broader rollout

5. **Chronic Disease Patient Protocol**
   - If chronic patients show higher no-show: Implement 3-touch reminder protocol (SMS at 7d, 2d, 1d pre-appointment)
   - Assign care coordinator for high-risk patients (diabetes + age 65+)

6. **Elderly Patient Engagement**
   - If 65+ cohort shows >25% no-show: Design large-text appointment confirmations
   - Offer phone call confirmations in addition to SMS

### Strategic Improvements (3+ Months)

7. **Reminder Channel Diversification**
   - If SMS alone shows plateau: Pilot WhatsApp reminders for mobile-registered patients
   - Test automated phone reminders for elderly cohort
   - **Logic:** Different channels may have differential effectiveness

8. **Socioeconomic Support Program**
   - If scholarship patients show no-show >25%: Establish transportation voucher pilot or childcare assistance
   - **Logic:** Remove barriers, improve attendance for vulnerable populations

9. **Real-Time Capacity Management**
   - Build dynamic overbooking model based on historical wait time, age, and SMS status
   - If long-wait elderly patients have >30% no-show, overbook by 30-40%

---

## Assumptions

1. **SMS Message Quality & Timing** – Assumption: SMS sent at optimal times (e.g., 24-48 hours pre-appointment). Reality Check: If SMS appears ineffective, audit timing and message clarity with patient surveys.

2. **Data Capture Accuracy** – Assumption: No-show status accurately recorded. Reality Check: Verify no data entry errors; confirm if late arrivals (>15min) marked differently.

3. **Temporal Stability** – Assumption: 2016 attendance patterns still applicable in current operations. Reality Check: Seasonal factors, COVID-related shifts, new transportation options may have changed behavior.

4. **Demographics as Unchanging** – Assumption: Age, chronic condition status, neighborhood don't change during appointment booking. Reality Check: For multi-month bookings, patient health could change.

5. **SMS as Marker, Not Cause** – Assumption: SMS is likely correlated with better systems, not causally improving attendance. Reality Check: Higher SMS receipt may indicate operational capacity or patient engagement rather than SMS creating attendance.

---

## Limitations

### Data & Scope Constraints

1. **Single Geographic Region** – Dataset represents Brazilian healthcare system (2016); findings may not generalize to different countries, healthcare systems, or years post-2016.

2. **No Causality** – Analysis shows **correlation, not causality**. If SMS-receiving patients have lower no-show rate, this could mean:
   - SMS reduces no-shows (causal), OR
   - Engaged patients are more likely to sign up for SMS (selection bias), OR
   - SMS recipients are higher SES with better transportation (confounding)
   - **Implication:** Recommendations are operational hypotheses, not proven interventions.

3. **Missing Operational Variables** – No appointment duration, procedure type, reason for scheduling, or patient distance from facility (all likely strong predictors).

4. **Temporal Snapshot** – Analysis is cross-sectional (single time period, no before/after); cannot measure impact of interventions or changes over time.

5. **Cannot Quantify Individual Risk** – Analysis provides group-level insights; cannot predict which specific patient will no-show.

### Analytical Approach Limitations

6. **Summary Statistics Only** – No advanced modeling; relies on basic counts, percentages, visualizations.

7. **Chi-Square Test Assumptions** – Chi-square sensitive to sample size; with 110K records, even tiny effects show as "statistically significant." **Implication:** Statistical significance ≠ operational significance.

---

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.7+ | Data analysis programming language |
| **pandas** | 1.x | Data loading, cleaning, feature engineering |
| **NumPy** | 1.x | Numerical calculations |
| **matplotlib** | 3.x | Static visualizations and charts |
| **seaborn** | 0.11+ | Statistical visualization and aesthetics |
| **SciPy** | 1.x | Chi-square statistical testing |
| **Jupyter Notebook** | - | Interactive analysis environment |

---

## How to Run the Notebook

### Prerequisites
- Python 3.7 or later
- Jupyter Notebook or JupyterLab
- GitHub repository cloned or files downloaded locally

### Setup Instructions

1. **Navigate to Project Directory**
   ```bash
   cd PHARMA_PROJECT
   ```

2. **Install Required Packages** (if not already installed)
   ```bash
   pip install pandas numpy matplotlib seaborn scipy jupyter
   ```
   
   Or install from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Data Files**
   - Ensure `Data.csv` is in the same directory as the notebook
   - The notebook uses relative file paths; file placement matters

4. **Launch Jupyter Notebook**
   - Using Jupyter Notebook:
     ```bash
     jupyter notebook healthcare_eda_1.ipynb
     ```
   - Using JupyterLab:
     ```bash
     jupyter lab healthcare_eda_1.ipynb
     ```

5. **Run Cells Sequentially**
   - Click **Cell → Run All** OR
   - Click through cells one-by-one from top (recommended for inspection)
   - Outputs render inline below each cell
   - Execution time: ~3-5 minutes for full notebook

### Execution Steps (In Notebook)

| Step | Purpose | Expected Output |
|------|---------|-----------------|
| **1. Imports & Data Load** | Load libraries and raw data | 110,527 rows × [columns] |
| **2. Data Type Inspection** | Check formats (dates as strings?) | DataFrame info with dtypes |
| **3. Date Parsing** | Convert date strings to datetime | Parsed dates ready for calculations |
| **4. Feature Engineering** | Create age_group, wait_category, etc. | New columns added to DataFrame |
| **5. Summary Statistics** | Baseline metrics | Summary table |
| **6. Demographic Analysis** | Age, condition, scholarship patterns | Charts: no-show % by segment |
| **7. Appointment Analysis** | Wait time, day-of-week, location patterns | Charts: no-show % by timing/location |
| **8. SMS Effectiveness** | Reminder impact assessment | Comparison: SMS vs. no SMS |
| **9. Statistical Tests** | Chi-square validation | p-values, significance indicators |
| **10. Recommendations** | Summary insights and actionable steps | Bullet-point recommendations |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **FileNotFoundError: Data.csv not found** | Verify Data.csv is in same folder as notebook |
| **ModuleNotFoundError: pandas** | Run `pip install pandas` in terminal |
| **Kernel not responding** | Restart kernel: Kernel → Restart Kernel |
| **Visualizations not showing** | Ensure `%matplotlib inline` is in first code cell |

---

### File Descriptions

| File | Description |
|------|-------------|
| **README.md** | Project overview, methodology, insights, and instructions |
| **requirements.txt** | Python dependencies; use `pip install -r requirements.txt` |
| **Data.csv** | 110,527 healthcare appointments from Brazil (2016); publicly available from Kaggle |
| **healthcare_eda_1.ipynb** | Jupyter notebook with full analysis: data cleaning, feature engineering, visualizations, statistical tests |
| **pharma_sales_analysis.ipynb** | Separate project (not related to healthcare analysis) |
| **pharma_dataset.csv** | Separate project dataset |

### Notes
- This project is **notebook-first**: all analysis is contained in the Jupyter notebook
- No separate Python scripts or modules
- All visualization outputs render inline in notebook
- Reproducible: running the notebook from step 1 regenerates all outputs

---

## Author & Project Info

**Project Title:** Medical Appointment No-Show Analysis  
**Type:** Exploratory Data Analysis (Operational Insights)  
**Dataset:** Kaggle Medical Appointment No-Show (Brazil, 2016)  
**Analysis Date:** February 2026  
**Focus:** Healthcare operations optimization, not predictive modeling  

**Suitable For:**
- Healthcare Operations teams
- Data Analyst portfolios
- Healthcare analytics interviews
- Operational dashboarding groundwork

---

## License & Attributions

**Data Source:** [Kaggle - Medical Appointment No-shows](https://www.kaggle.com/joniarroba/noshowappointments) (Public Dataset)

**Original Dataset Credit:** Kaggle Community  
**Analysis & Report:** Original work for portfolio purposes

---

**Last Updated:** February 23, 2026


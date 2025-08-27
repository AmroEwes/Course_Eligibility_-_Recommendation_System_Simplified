import pandas as pd
import ast
import streamlit as st
import numpy as np

# Sample DataFrame
def create_sample_data():
    sample_data = pd.DataFrame({
        'Student_ID': [112255,112255, 24856,24856],
        'Semester': [2202,2202, 2302,2302],
        'College': ['CBA','CBA','CAS', 'CAS'],
        'Passed Credits': [30,30,45, 45],
        'Student_Level': [1,1,2, 2],
        'Program': ['Accounting','Accounting','Computer Science', 'Computer Science'],
        'Major': ['Accounting','Accounting','Computer Science', 'Computer Science'],
        'Course_ID': ['ACCT201','MATH111', 'CSC123','ENGL110'],
        'GRADE': ['A','A-', 'B','B+']
    })
    return sample_data

st.set_page_config(page_title="Course Eligibility & Recommendation System", layout="wide")
st.image("Grad_Icon.png",width=400)
navigation = st.sidebar.radio("Go To", ["User Guide", "Course Eligibility and Recommendation System","Quick Check"])

# Add custom CSS to the Streamlit app
st.markdown(
    """
    <style>
    /* Ensure the font-family applies to all text elements */
    @font-face {
        font-family: 'Times New Roman';
        src: url('https://fonts.cdnfonts.com/s/15292/Times_New_Roman.woff') format('woff');
    }
    body, div, p, h1, h2, h3, h4, h5, h6, span, td, th, li, label, input, button, select, textarea, .stMarkdown, .stTextInput, .stTextArea, .stRadio, .stCheckbox, .stSelectbox, .stMultiSelect, .stButton, .stSlider, .stDataFrame, .stTable, .stExpander, .stTabs, .stAccordion, .stDownloadButton {
        font-family: 'Times New Roman', serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def st_data_cleaning(st_enrollment_data, transfer_credit_data):
    ac_st_enrollment_data = st_enrollment_data
    tc_data = transfer_credit_data
    
    # Rename columns
    rename_columns_dict = {"EMPLID": "Student_ID", "STRM": "Semester", "Course": "Course_ID",
                           "Level": "Student_Level", "Plan": "Major",
                           "COURSE": "Course_ID", "chosen_semester": "Semester",
                          "STUDENT_ID" : "Student_ID","TRANSFER_TERM":"Semester",
                          "UNT_TRNSFR":"CREDITS","CUM_GPA":"GPA"}
    tc_data.rename(columns=rename_columns_dict, inplace=True)
    ac_st_enrollment_data.rename(columns=rename_columns_dict, inplace=True)
    ac_st_enrollment_data = ac_st_enrollment_data.fillna(0)
    ac_st_enrollment_data = ac_st_enrollment_data.astype(str)
    tc_data = tc_data.astype(str)
    
    # Strip whitespace from all columns
    for df in [tc_data, ac_st_enrollment_data]:
        for column in df.columns:
            df[column] = df[column].str.strip()
            
    # Convert relevant columns to appropriate types
    ac_st_enrollment_data["Student_Level"] = ac_st_enrollment_data["Student_Level"].str.extract('(\d+)', expand=False)
    ac_st_enrollment_data = ac_st_enrollment_data.dropna(subset=["Student_Level"]).copy()
    ac_st_enrollment_data["Semester"] = ac_st_enrollment_data["Semester"].astype(int)
    ac_st_enrollment_data["Student_Level"] = ac_st_enrollment_data["Student_Level"].astype(int)
    ac_st_enrollment_data["CREDITS"] = ac_st_enrollment_data["CREDITS"].astype(float)
    ac_st_enrollment_data["Passed Credits"] = ac_st_enrollment_data["Passed Credits"].astype(float)
    ac_st_enrollment_data["GPA"] = ac_st_enrollment_data["GPA"].astype(float)
    ac_st_enrollment_data["MPA"] = ac_st_enrollment_data["MPA"].astype(float)
    tc_data["Semester"] = tc_data["Semester"].astype(int)
    tc_data["CREDITS"] = tc_data["CREDITS"].astype(float)
    
    ac_st_enrollment_data["CREDITS"] = ac_st_enrollment_data["CREDITS"].astype(int)
    ac_st_enrollment_data["Passed Credits"] = ac_st_enrollment_data["Passed Credits"].astype(int)
    tc_data["CREDITS"] = tc_data["CREDITS"].astype(int)
    
    max_semester_index = ac_st_enrollment_data.groupby('Student_ID')['Semester'].idxmax()
    latest_major_df = ac_st_enrollment_data.loc[max_semester_index, ['Student_ID', 'Semester',
                                                              "College", "Program", 'Major']]
    
    
    ac_st_enrollment_data = pd.merge(ac_st_enrollment_data.drop(columns=["College", "Program", 'Major']),
                          latest_major_df[["Student_ID", "College", "Program", 'Major']],
                          left_on="Student_ID", right_on="Student_ID", how='inner')

    tc_data = pd.merge(tc_data, latest_major_df[["Student_ID", "College", "Program", 'Major']],
                           left_on="Student_ID", right_on="Student_ID", how='inner')
    
    # Define a function to determine the chosen semester
    def determine_chosen_semester(row):
        if row['Semester'] == row['min']:
            return row['Semester']
        else:
            return row['min']
        
    semester_stats = ac_st_enrollment_data.groupby('Student_ID')['Semester'].agg(['min', 'max']).reset_index()
    tc_data = tc_data.merge(semester_stats, on='Student_ID', how='left')
    
    tc_data['chosen_semester'] = tc_data.apply(determine_chosen_semester, axis=1)
    tc_data = tc_data.drop(columns=["min", "max", "Semester", "SUBJECT","CATALOG_NBR"])
    tc_data.rename(columns=rename_columns_dict, inplace=True)
    grouped_data = ac_st_enrollment_data.groupby(['Student_ID', 'Semester']).agg({
        'Student_Level': 'first',
        'ADMIT_TERM': 'first',
        'Passed Credits': 'first',
        'Status': 'first',
        'GPA': 'first',
        'MPA': 'first'
    }).reset_index()
    
    tc_data = pd.merge(tc_data, grouped_data, on=['Student_ID', 'Semester'], how='inner')

    # Select and reorder columns
    tc_data = tc_data[['Student_ID', 'Semester', 'Status', 'Student_Level',
                       'Course_ID',"CREDITS", 'College', 'Program', 'Major', 'ADMIT_TERM',
                       'Passed Credits', 'GPA', 'MPA']]

    # Filter out unwanted records
    values_to_delete = ['Visit', 'Non-Degree', 'Undeclared - English',
                        'FA', 'F', 'I', 'S', 'NP', 'WA']

    ac_st_enrollment_data = ac_st_enrollment_data[~ac_st_enrollment_data["Major"].isin(values_to_delete)]
    ac_st_enrollment_data = ac_st_enrollment_data[['Student_ID', 'Semester','GRADE', 'Status', 'Student_Level',
                                     'Course_ID',"CREDITS", 'College', 'Program', 'Major', 'ADMIT_TERM',
                                     'Passed Credits', 'GPA', 'MPA']]
    
    # Combine data
    combined_data = pd.concat([ac_st_enrollment_data, tc_data], axis=0)
    combined_data["Major"] = combined_data['Major'].replace('Radio / TV', 'Digital Media Production')
    
    # Identify the latest semester for each student
    latest_semester = combined_data.groupby('Student_ID')['Semester'].max().reset_index()
    latest_semester.columns = ['Student_ID', 'Latest_Semester']

    # Merge this information back with the original dataframe
    combined_data = pd.merge(combined_data, latest_semester, on='Student_ID')

    # Filter rows for each student where the semester is their latest semester
    latest_semester_data = combined_data[combined_data['Semester'] == combined_data['Latest_Semester']]

    # Extract the passed credits for each student from the latest semester data
    latest_semester_passed_credits = latest_semester_data[['Student_ID', 'Passed Credits']].drop_duplicates()
    # Summing the CREDITS for the latest semester
    latest_semester_credits_sum = latest_semester_data.groupby('Student_ID')['CREDITS'].sum().reset_index()
    latest_semester_credits_sum.columns = ['Student_ID', 'Latest_Semester_Credits']


    previous_semesters_data = combined_data[combined_data['Semester'] != combined_data['Latest_Semester']]
    # Identify the latest semester for each student
    latest_semester_previous = previous_semesters_data.groupby('Student_ID')['Semester'].max().reset_index()
    latest_semester_previous.columns = ['Student_ID', 'Latest_Semester_Previous']

    # Merge this information back with the original dataframe
    combined_data = pd.merge(combined_data, latest_semester_previous, on='Student_ID',how = "left")
    combined_data = combined_data.fillna(0)
    combined_data["Latest_Semester_Previous"] = combined_data["Latest_Semester_Previous"].astype(int)

    # Filter rows for each student where the semester is their latest semester
    latest_semester_previous_data = combined_data[combined_data['Semester'] == combined_data['Latest_Semester_Previous']]

    # Extract the passed credits for each student from the latest semester data
    latest_semester_previous_passed_credits = latest_semester_previous_data[['Student_ID', 'Passed Credits']].drop_duplicates()

    latest_semester_passed_credits = latest_semester_passed_credits.rename(columns={"Passed Credits":"Passed_Credits_Latest"})
    latest_semester_previous_passed_credits = latest_semester_previous_passed_credits.rename(columns={"Passed Credits":"Passed_Credits_Previous"})

    total_pcr_previous_latest = pd.merge(latest_semester_passed_credits,latest_semester_previous_passed_credits, on='Student_ID',how = "left")
    total_pcr_previous_latest = total_pcr_previous_latest.fillna(0)
    total_pcr_previous_latest["Passed_Credits_Previous"] = total_pcr_previous_latest["Passed_Credits_Previous"].astype(int)

    total_pcr_previous_latest = pd.merge(latest_semester_credits_sum, total_pcr_previous_latest, on='Student_ID')
    total_pcr_previous_latest['Incoming_PCR'] = total_pcr_previous_latest.apply(
        lambda row: row['Latest_Semester_Credits'] + row['Passed_Credits_Latest'] 
        if row['Passed_Credits_Previous'] == row['Passed_Credits_Latest'] 
        else row['Passed_Credits_Latest'],
        axis=1)
    combined_data = pd.merge(combined_data, total_pcr_previous_latest[['Student_ID', 'Incoming_PCR']], on='Student_ID')
    combined_data['Incoming_PCR'] = combined_data.apply(
        lambda row: 0 if row['Semester'] != row['Latest_Semester'] else row['Incoming_PCR'],
        axis=1)
    combined_data = combined_data.drop(columns=["Latest_Semester_Previous","Latest_Semester"])
    
    return combined_data

# Eligibility Functions
def is_eligible(course, taken_courses, prerequisites):
    prereqs = prerequisites.get(course, [])
    return all(prereq in taken_courses for prereq in prereqs)

# Consolidated universal eligibility function
def is_eligible_special(course, taken_courses, student_info, prerequisites, conditions):
    prereqs = prerequisites.get(course, [])
    condition = conditions.get(course, "")

    major = student_info.get('Major', '')
    program = student_info.get('Program', '')
    college = student_info.get('College', '')
    level = student_info.get('Student_Level', 0)
    passed_credits = student_info.get('Passed Credits', 0)
    incoming_pcr = int(student_info.get('Incoming_PCR', 0))

    if condition == "OR":
        return any(prereq in taken_courses for prereq in prereqs)
    elif condition == "AND":
        return all(prereq in taken_courses for prereq in prereqs)
    elif condition == "AND_NOT_CS":
        return all(prereq in taken_courses for prereq in prereqs) and major != "Computer Science"
    elif condition == "OR_AND_NOT_CS":
        return any(prereq in taken_courses for prereq in prereqs) and major != "Computer Science"
    elif condition == "Credits":
        return passed_credits >= 81 or incoming_pcr >= 81
    elif condition == "Credits_College":
        return (passed_credits >= 81 and college == "CBA") or (incoming_pcr >= 81 and college == "CBA")
    elif condition == "AND_OR":
        return prereqs and prereqs[0] in taken_courses and any(prereq in taken_courses for prereq in prereqs[1:])
    elif condition == "AND_OR_2":
        return prereqs and all(prereq in taken_courses for prereq in prereqs[:2]) and any(prereq in taken_courses for prereq in prereqs[2:])
    elif condition == "AND_OR_3":
        return prereqs and any(prereq in taken_courses for prereq in prereqs[:2]) and all(prereq in taken_courses for prereq in prereqs[2:])
    elif condition == "OR_AND":
        return prereqs and (all(prereq in taken_courses for prereq in prereqs[:2]) or any(prereq in taken_courses for prereq in prereqs[2:]))
    elif condition == "AND_3_Courses":
        return prereqs and all(prereq in taken_courses for prereq in prereqs[:2]) and sum(prereq in taken_courses for prereq in prereqs[2:]) >= 3
    elif condition == "Any_Two":
        return sum(prereq in taken_courses for prereq in prereqs) >= 2
    elif condition == "Any_Three":
        return sum(prereq in taken_courses for prereq in prereqs) >= 3
    elif condition == "AND_Senior":
        return all(prereq in taken_courses for prereq in prereqs) and level == 4
    elif condition == "Senior_AND_CBA":
        return level == 4 and college == "CBA"
    elif condition == "Junior_AND_Major_ACC":
        return level == 3 and major == "Accounting"
    elif condition == "AND_Major_ACC":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Accounting"
    elif condition == "AND_Major_FIN":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Finance"
    elif condition == "Senior_AND_Major_FIN":
        return level == 4 and major == "Finance"
    elif condition == "AND_Major_CS":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Computer Science"
    elif condition == "Junior_CS":
        return level == 3 and major == "Computer Science"
    elif condition == "Senior_CS":
        return level == 4 and major == "Computer Science"
    elif condition == "OR_CS":
        return any(prereq in taken_courses for prereq in prereqs) and major == "Computer Science"
    elif condition == "AND_College":
        return all(prereq in taken_courses for prereq in prereqs) and college == "CEA"
    elif condition == "AND_College_OR":
        return all(prereq in taken_courses for prereq in prereqs) and (major == "Computer Science" or college == "CEA")
    elif condition == "OR_AND_College_OR":
        return any(prereq in taken_courses for prereq in prereqs) and (major == "Computer Science" or college == "CEA")
    elif condition == "Junior_ECOM":
        return level == 3 and program == "Computer Engineering"
    elif condition == "Senior_ECOM":
        return level == 4 and program == "Computer Engineering"
    elif condition == "Senior_And_Major_MG_IB":
        return level == 4 and (major == "International Business" or major == "Mgmt & Organizational Behavior")
    elif condition == "Junior_And_Major_IB":
        return level == 3 and major == "International Business"
    elif condition == "Junior_And_Major_MOB":
        return level == 3 and major == "Mgmt & Organizational Behavior"
    elif condition == "AND_Major_MG_IB":
        return all(prereq in taken_courses for prereq in prereqs) and (major == "International Business" or major == "Mgmt & Organizational Behavior")
    elif condition == "AND_Major_MG_IB_MRKT":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["International Business", "Mgmt & Organizational Behavior", "Marketing"]
    elif condition == "AND_Major_MG_IB_MRKT_MIS":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["International Business", "Mgmt & Organizational Behavior", "Marketing", "Management Information Systems"]
    elif condition == "Senior_AND_Major_MRKT":
        return level == 4 and major == "Marketing"
    elif condition == "Junior_AND_Major_MRKT":
        return level == 3 and major == "Marketing"
    elif condition == "AND_Major_MRKT":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Marketing"
    elif condition == "Senior_AND_Major_MIS":
        return level == 4 and major == "Management Information Systems"
    elif condition == "Junior_AND_Major_MIS":
        return level == 3 and major == "Management Information Systems"
    elif condition == "AND_Major_MIS":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Management Information Systems"
    elif condition == "AND_Credits_MIS_CS":
        return all(prereq in taken_courses for prereq in prereqs) and (passed_credits < 45) and major in ["Management Information Systems", "Computer Science"]
    elif condition == "AND_UENG":
        return all(prereq in taken_courses for prereq in prereqs) and program == "English"
    elif condition == "Senior_Lingusitics":
        return level == 4 and major == "Eng- Linguistics - Translation"
    elif condition == "Senior_AND_UENG":
        return level == 4 and program == "English"
    elif condition == "AND_LIN_LIT":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["Eng- Linguistics - Translation", "English Literature"]
    elif condition == "AND_EDU":
        return all(prereq in taken_courses for prereq in prereqs) and major == "English Education"
    elif condition == "AND_MCOM":
        return all(prereq in taken_courses for prereq in prereqs) and program == "Mass Communication"
    elif condition == "OR_MCOM":
        return any(prereq in taken_courses for prereq in prereqs) and program == "Mass Communication"
    elif condition == "AND_Credits_MCOM":
        return all(prereq in taken_courses for prereq in prereqs) and ((passed_credits >= 54 or incoming_pcr >= 54) and program == "Mass Communication")
    elif condition == "AND_Credits_MCOM_2":
        return all(prereq in taken_courses for prereq in prereqs) and ((passed_credits >= 60 or incoming_pcr >= 60) and program == "Mass Communication")
    elif condition == "AND_OR_PR":
        return (all(prereq in taken_courses for prereq in prereqs[:3]) and major == "Public relations & Advertising") or all(prereq in taken_courses for prereq in prereqs[4:])
    elif condition == "AND_OR_Junior_Program":
        return prereqs and prereqs[0] in taken_courses and any(prereq in taken_courses for prereq in prereqs[1:]) and level == 3 and program == "Mass Communication"
    elif condition == "Junior_Program":
        return level == 3 and program == "Mass Communication"
    elif condition == "Senior_MCOM":
        return level == 4 and program == "Mass Communication"
    elif condition == "AND_Junior":
        return level == 3 and all(prereq in taken_courses for prereq in prereqs)
    elif condition == "AND_Junior_Program":
        return level == 3 and all(prereq in taken_courses for prereq in prereqs) and program == "Mass Communication"
    elif condition == "OR_AND_Program_OR":
        return any(prereq in taken_courses for prereq in prereqs) and program in ["Mass Communication", "English"]
    elif condition == "AND_NOT_ENGLISH":
        return all(prereq in taken_courses for prereq in prereqs) and program != "English"
    elif condition == "OR_AND":
        return all(prereq in taken_courses for prereq in prereqs[:2]) or any(prereq in taken_courses for prereq in prereqs[2:])
    elif condition == "AND_Credits_MIS_CS":
        return (
        all(prereq in taken_courses for prereq in prereqs) and
        ((passed_credits < 45 and major in ["Management Information Systems", "Computer Science"]) or
        (incoming_pcr < 45 and major in ["Management Information Systems", "Computer Science"])))
    else:
        return False


def is_eligible_special_(course, taken_courses, student_info, prerequisites, conditions):
    prereqs = prerequisites.get(course, [])
    condition = conditions.get(course, "")

    major = student_info.get('Major', '')
    program = student_info.get('Program', '')
    college = student_info.get('College', '')
    level = student_info.get('Student_Level', 0)
    passed_credits = student_info.get('Passed Credits', 0)
    incoming_pcr = int(student_info.get('Incoming_PCR', 0))

    if condition == "OR":
        return any(prereq in taken_courses for prereq in prereqs)

    elif condition == "AND_NOT_CS":
        return all(prereq in taken_courses for prereq in prereqs) and major != "Computer Science"

    elif condition == "OR_AND_NOT_CS":
        return any(prereq in taken_courses for prereq in prereqs) and major != "Computer Science"

    elif condition == "AND_Senior":
        return all(prereq in taken_courses for prereq in prereqs) and level == 4

    elif condition == "AND_Major_ACC":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Accounting"

    elif condition == "AND_Major_FIN":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Finance"

    elif condition == "AND_Major_MG_IB":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["International Business", "Mgmt & Organizational Behavior"]

    elif condition == "AND_Major_MG_IB_MRKT":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["International Business", "Mgmt & Organizational Behavior", "Marketing"]

    elif condition == "AND_Major_MG_IB_MRKT_MIS":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["International Business", "Mgmt & Organizational Behavior", "Marketing", "Management Information Systems"]

    elif condition == "AND_Major_MRKT":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Marketing"

    elif condition == "AND_Major_MIS":
        return all(prereq in taken_courses for prereq in prereqs) and major == "Management Information Systems"

    elif condition == "AND_Credits_MIS_CS":
        return (
            all(prereq in taken_courses for prereq in prereqs) and
            (
                (passed_credits < 45 and major in ["Management Information Systems", "Computer Science"]) or
                (incoming_pcr < 45 and major in ["Management Information Systems", "Computer Science"])
            )
        )

    elif condition == "AND_MCOM":
        return all(prereq in taken_courses for prereq in prereqs) and program == "Mass Communication"

    elif condition == "OR_MCOM":
        return any(prereq in taken_courses for prereq in prereqs) and program == "Mass Communication"

    elif condition == "AND_Credits_MCOM":
        return all(prereq in taken_courses for prereq in prereqs) and passed_credits >= 54 and program == "Mass Communication"

    elif condition == "AND_Credits_MCOM_2":
        return all(prereq in taken_courses for prereq in prereqs) and passed_credits >= 60 and program == "Mass Communication"

    elif condition == "OR_CS":
        return any(prereq in taken_courses for prereq in prereqs) and major == "Computer Science"

    elif condition == "AND_College_OR":
        return all(prereq in taken_courses for prereq in prereqs) and (major == "Computer Science" or college == "CEA")

    elif condition == "OR_AND_College_OR":
        return any(prereq in taken_courses for prereq in prereqs) and (major == "Computer Science" or college == "CEA")

    elif condition == "AND_Junior":
        return level == 3 and all(prereq in taken_courses for prereq in prereqs)

    elif condition == "AND_Junior_Program":
        return level == 3 and all(prereq in taken_courses for prereq in prereqs) and program == "Mass Communication"

    elif condition == "OR_AND_Program_OR":
        return any(prereq in taken_courses for prereq in prereqs) and program in ["Mass Communication", "English"]

    elif condition == "AND_UENG":
        return all(prereq in taken_courses for prereq in prereqs) and program == "English"

    elif condition == "AND_LIN_LIT":
        return all(prereq in taken_courses for prereq in prereqs) and major in ["Eng- Linguistics - Translation", "English Literature"]

    elif condition == "AND_EDU":
        return all(prereq in taken_courses for prereq in prereqs) and major == "English Education"

    elif condition == "AND_College":
        return all(prereq in taken_courses for prereq in prereqs) and college == "CEA"

    elif condition == "AND_NOT_ENGLISH":
        return all(prereq in taken_courses for prereq in prereqs) and program != "English"

    else:
        return False


# Helper Functions from provided logic
def combine_eligible_courses(df1, df2):
    if df1.shape != df2.shape:
        raise ValueError("Dataframes do not have the same shape.")
    
    if list(df1.columns) != list(df2.columns):
        raise ValueError("Dataframes do not have the same headers.")
    
    combined_data = []
    for index, row in df1.iterrows():
        combined_row = row.copy()
        combined_courses = list(set(row['Eligible_Courses'] + df2.loc[index, 'Eligible_Courses']))
        combined_row['Eligible_Courses'] = combined_courses
        combined_data.append(combined_row)
    
    combined_df = pd.DataFrame(combined_data)
    
    return combined_df

def find_course_combinations(student_courses, requisites_data):
    combinations = []
    for _, row in requisites_data.iterrows():
        requisites_list = row['REQUISITES_LIST']
        course_id = row['Course_ID']
        if all(course in student_courses for course in requisites_list):
            combination = requisites_list + [course_id]
            combinations.append(combination)
    return combinations

def create_combined_courses(row, co):
    eligible_courses = row['Eligible_Courses']
    combined_courses = eligible_courses[:]
    co_requisite_courses = []
    combinations = find_course_combinations(eligible_courses, co)
    for combination in combinations:
        combined_courses += combination
        co_requisite_courses.append(combination)
    row['Co_Requisite_Courses'] = co_requisite_courses
    row['Eligible_Courses_CO'] = list(set(combined_courses))
    return row

def find_additional_eligibilities(courses, taken_courses, prerequisites):
    additional_eligibilities = set()
    for course in courses:
        hypothetical_courses = taken_courses.copy()
        hypothetical_courses.add(course)
        for c in prerequisites.keys():
            if is_eligible(c, hypothetical_courses, prerequisites) and c not in hypothetical_courses:
                additional_eligibilities.add(c)
    return list(additional_eligibilities)

def find_additional_eligibilities_special(courses, taken_courses, student_info, prerequisites_special, conditions, is_eligible_special):
    additional_eligibilities = set()
    hypothetical_courses = taken_courses.copy()
    for course in courses:
        hypothetical_courses.add(course)
        for c in prerequisites_special.keys():
            if is_eligible_special(c, hypothetical_courses, student_info, prerequisites_special, conditions) and c not in hypothetical_courses:
                additional_eligibilities.add(c)
    return list(additional_eligibilities)

# Function to remove matches
def remove_matches(row):
    eligible_courses = set(row["Eligible_Courses_CO"])
    course_id = set(row["Course_ID"])
    unmatched_courses = eligible_courses - course_id  # Set difference to find unmatched courses
    return list(unmatched_courses)  # Return as list for compatibility

# Function to process each row based on multiple course categories
def process_row(row):
    course_data = [
        (['MATH100','MATH131','MATH132','MATH231','MATH140','MATH221','MATH211',
          'MATH330','MATH111','MATH130','MATH121','MATH400','MATH331','MATH342',
          'MATH232','MATH122','MATH120'],
         ['MATH094','MATH095','MATH096','MATH098']),
        
        (['ENGL100','ENGL110','ENGL112'],
         ['ENGL097','ENGL098']),
         (['MATH096'],
         ['MATH095','MATH094']),
         (['MATH098'],
         ['MATH096','MATH095','MATH094'])
    ]

    for check_courses, remove_courses in course_data:
        if any(course in check_courses for course in row['Course_ID']):
            row['Eligible_Courses_CO'] = [course for course in row['Eligible_Courses_CO'] if course not in remove_courses]
    
    return row

def find_best_courses(group):
    sorted_courses = group.sort_values(by='Course_Score', ascending=False)
    return sorted_courses['Eligible_Courses_CO'].tolist()[:5]

def find_best_courses_v2(group):
    sorted_courses = group.sort_values(by='Course_Score', ascending=False)
    return sorted_courses['Eligible_Courses_CO'].tolist()[:5]

# Function to normalize the scores per student for each eligible course using Max Normalization
def normalize_by_student(group):
    # Normalize the Course Score by Max Normalization
    if group['Course_Score'].max() > 0:  # Avoid division by zero
        group['Normalized_Course_Score'] = group['Course_Score'] / group['Course_Score'].max()
    else:
        group['Normalized_Course_Score'] = 0  # Assign a default value if all scores are the same
    
    # Normalize the Remaining Weight Score by Max Normalization
    if group['Remaining_Courses_Weight_Score'].max() > 0:
        group['Normalized_Remaining_Courses_Weight'] = group['Remaining_Courses_Weight_Score'] / group['Remaining_Courses_Weight_Score'].max()
    else:
        group['Normalized_Remaining_Courses_Weight'] = 0  # Assign a default value if all scores are the same

    # Normalize the Course Level by Max Normalization and invert it
    if group['Course_Level'].max() > 0:
        group['Normalized_Course_Level'] = 1 - (group['Course_Level'] / group['Course_Level'].max())
    else:
        group['Normalized_Course_Level'] = 0  # Assign a default value if all levels are the same

    return group

def find_best_courses_cea_v2(group):
    sorted_courses = group.sort_values(by='Final_Score', ascending=False)
    return sorted_courses['Eligible_Courses_CO'].tolist()[:7]


def process_data_generic(st_hist_data,major_data, requirements_weights_path,major_name, major_code,college_groups):
    
    values_to_delete = ['FA', 'F', 'I', 'S', 'NP', 'WA']
    failed_grades = ['F','FA','NP']
    failed_data = st_hist_data[st_hist_data["GRADE"].isin(failed_grades)]
    st_hist_data = st_hist_data[~st_hist_data["GRADE"].isin(values_to_delete)]
    
    # Filtering and Sorting Data
    failed_data = failed_data[failed_data['Major'] == major_name]
    failed_data = failed_data.sort_values(by=['Student_ID', 'Semester'])

    grouped_data_failed = failed_data.groupby(['Student_ID'])['Course_ID'].apply(list).reset_index()
    
    # Filtering and Sorting Data
    st_enrollment_data = st_hist_data[st_hist_data['Major'] == major_name]
    st_enrollment_data = st_enrollment_data.sort_values(by=['Student_ID', 'Semester'])

    major = major_data["All_Courses"]
    courses_co = major_data["CO_Courses"]
    
    major["AREA_OF_STUDY"] = major["AREA_OF_STUDY"].fillna("NA")
    # Dropping records where AREA_OF_STUDY is 'N' and COURSE_OF_STUDY is 'Z'
    major_filtered = major[~((major['AREA_OF_STUDY'] == 'NA') & (major['COURSE_OF_STUDY'] == 'Z'))]
    
    major_filtered = major_filtered.copy()
    # Apply replacements directly to the specific columns to avoid SettingWithCopyWarning
    #major_filtered['AREA_OF_STUDY'] = major_filtered['AREA_OF_STUDY'].replace("NA","GE")
    #major_filtered['COURSE_OF_STUDY'] = major_filtered['COURSE_OF_STUDY'].replace("N","E")
    
    # Defining the major lists
    majors_list = college_groups
    all_courses_majors = major_filtered[(major_filtered['Major'].isin(majors_list)) & (major_filtered['COURSE_OF_STUDY'].isin(['R', 'RE','E']))]   
    
    list_conditions = ['-', 'ONE_COURSE']

    list_ = all_courses_majors[all_courses_majors['Condition'].isin(list_conditions)]
    special_cases = all_courses_majors[~all_courses_majors['Condition'].isin(list_conditions)]
    co = courses_co[courses_co['Major'].isin(majors_list)]
    
    courses_list = list_[list_["Major"] == major_code]
    courses_special_cases = special_cases[special_cases["Major"] == major_code]
    courses_co = co[co["Major"] == major_code]

    # Process 'REQUISITES_LIST'
    courses_co = courses_co.copy()
    courses_co.loc[:, 'REQUISITES_LIST'] = courses_co['REQUISITES_LIST'].apply(ast.literal_eval)

    # CAS Courses
    college_courses = major_filtered[major_filtered['Major'].isin(majors_list)]
    major_courses = college_courses[college_courses["Major"] == major_code]
    
    grouped_data = st_enrollment_data.groupby(['Student_ID'])['Course_ID'].apply(list).reset_index()

    # Merge dataframes
    merged_df = grouped_data_failed.merge(grouped_data, on=['Student_ID'], how='outer', suffixes=('_failed', '_all'))
    # Replace NaN with empty lists to avoid errors
    merged_df['Course_ID_all'] = merged_df['Course_ID_all'].apply(lambda x: x if isinstance(x, list) else [])
    merged_df['Course_ID_failed'] = merged_df['Course_ID_failed'].apply(lambda x: x if isinstance(x, list) else [])

    merged_df['Failed_Courses'] = merged_df.apply(
        lambda row: list(set(row['Course_ID_failed']) - set(row['Course_ID_all'])),
        axis=1)
    # Keep only relevant columns
    merged_df = merged_df[['Student_ID', 'Failed_Courses']]

    # Extract Accounting specific requirements and weights from respective DataFrames
    requirements_df = pd.read_excel(requirements_weights_path,sheet_name="requirements")
    weights_df = pd.read_excel(requirements_weights_path,sheet_name="weights")
    requirements = requirements_df[requirements_df["Major"] == major_name]
    requirements_ = requirements.pivot_table(index="Major",columns="AREA_OF_STUDY",values ='Required_Courses' ,aggfunc='sum',fill_value=0).reset_index()
    weights = weights_df[weights_df["Major"] == major_name]

    student_courses = st_enrollment_data[["Student_ID", "Course_ID"]]

    # Map AREA_OF_STUDY and COURSE_OF_STUDY to comp_data
    student_courses = student_courses.merge(major_courses[['Course_ID', 'AREA_OF_STUDY', 'COURSE_OF_STUDY', "Course_Level"]],
                                            on='Course_ID', how='left').drop_duplicates()
    
    
    # Create summary DataFrames for taken courses
    student_progress = student_courses.groupby(['Student_ID', 'AREA_OF_STUDY']).size().reset_index(name='Total_Taken_Courses')
    # Get unique student IDs
    students = student_progress[['Student_ID']].drop_duplicates()
    # Cross join students with requirements
    cross = students.assign(key=1).merge(requirements.assign(key=1), on='key').drop('key', axis=1)
    student_progress = cross.merge(student_progress, on=['Student_ID', 'AREA_OF_STUDY'], how='left')
    # Fill missing courses with 0
    student_progress['Total_Taken_Courses'] = student_progress['Total_Taken_Courses'].fillna(0).astype(int)
    student_progress["Remaining_Courses"] = student_progress["Required_Courses"] - student_progress["Total_Taken_Courses"]
    student_progress["Remaining_Courses"] = student_progress["Remaining_Courses"].apply(lambda x: max(x, 0))

    free_comptive_taken_counts = student_courses[(student_courses['AREA_OF_STUDY'] == "GE") & (student_courses['COURSE_OF_STUDY'] == "E")].groupby('Student_ID').size().reset_index(name='Total_Free_comptives_Taken')

    student_progress.loc[student_progress["AREA_OF_STUDY"] == "FE", "Total_Taken_Courses"] = \
    student_progress.loc[student_progress["AREA_OF_STUDY"] == "FE"].merge(
        free_comptive_taken_counts, on="Student_ID", how="left"
    )["Total_Free_comptives_Taken"].fillna(
        student_progress.loc[student_progress["AREA_OF_STUDY"] == "FE", "Total_Taken_Courses"]
    ).values

    # Update progress by including the free elective data
    # Fill nulls with 0
    student_progress["Total_Taken_Courses"] = student_progress["Total_Taken_Courses"].fillna(0)
    student_progress["Required_Courses"] = student_progress["Required_Courses"].fillna(0)
    student_progress["Remaining_Courses"] = student_progress["Remaining_Courses"].fillna(0)

    # Calculate progress
    student_progress["Student_Progress"] = (student_progress["Total_Taken_Courses"] / student_progress["Required_Courses"]) * 100

    # Handle division by zero and cap at 100%
    student_progress["Student_Progress"].replace([np.inf, -np.inf], 100, inplace=True)
    student_progress["Student_Progress"].fillna(0, inplace=True)
    student_progress.loc[student_progress["Student_Progress"] > 100, "Student_Progress"] = 100
    summary_area_of_study_taken = student_progress.pivot_table(index="Student_ID", columns="AREA_OF_STUDY", values="Total_Taken_Courses", fill_value=0)
    #summary_area_of_study_taken = summary_area_of_study_taken.merge(free_comptive_taken_counts, on="Student_ID", how="left").fillna(0).rename(columns={"Total_Free_comptives_Taken": "FE"})

    # Create a copy of summary_area_of_study_taken to work on remaining courses calculation
    remaining_courses_df = summary_area_of_study_taken.copy()

    # Loop through each AREA_OF_STUDY and calculate remaining courses by subtracting from the requirements
    for column in remaining_courses_df.columns:
        if column in requirements['AREA_OF_STUDY'].values:
            required_courses = requirements.loc[requirements['AREA_OF_STUDY'] == column, 'Required_Courses'].values[0]
            remaining_courses_df[column] = required_courses - remaining_courses_df[column]
            remaining_courses_df[column] = remaining_courses_df[column].clip(lower=0)

    # Calculate weighted remaining courses
    weighted_remaining_courses_df = remaining_courses_df.copy()
    for column in weighted_remaining_courses_df.columns:
        if column in weights['AREA_OF_STUDY'].values:
            weight_value = weights.loc[weights['AREA_OF_STUDY'] == column, 'Weight'].values[0]
            weighted_remaining_courses_df[column] = weighted_remaining_courses_df[column] * weight_value

    # Prepare weighted remaining courses for merge
    weighted_remaining_courses_df = weighted_remaining_courses_df.reset_index().melt(id_vars=['Student_ID'],
                                                                                      var_name='AREA_OF_STUDY',
                                                                                      value_name='Remaining_Courses_Weight_Score')
    weighted_remaining_courses_df = weighted_remaining_courses_df[weighted_remaining_courses_df["AREA_OF_STUDY"] != "index"]

    # Eligibility Calculation for Standard and Special Cases
    prerequisites = courses_list.set_index('Course_ID')['REQUISITES_LIST'].apply(eval).to_dict()
    prerequisites_special = courses_special_cases.set_index('Course_ID')['REQUISITES_LIST'].apply(eval).to_dict()
    conditions = courses_special_cases.set_index('Course_ID')['Condition'].to_dict()

    final_results = []  # Standard eligibility results
    final_results_special = []  # Special eligibility results

    for student_id, group in st_enrollment_data.groupby('Student_ID'):
        cumulative_courses = set()
        for semester, semester_group in group.groupby('Semester'):
            taken_courses = set(semester_group['Course_ID'].tolist())
            cumulative_courses.update(taken_courses)

            # Determine Standard Eligible Courses
            student_info = semester_group.iloc[0].to_dict()
            eligible_courses = {course for course in prerequisites.keys() if all(req in cumulative_courses for req in prerequisites[course])}
            final_results.append({
                'Student_ID': student_id,
                'Semester': semester,
                'Major': student_info['Major'],
                'College': student_info['College'],
                'Program': student_info['Program'],
                'Passed Credits': student_info['Passed Credits'],
                'Student_Level': student_info['Student_Level'],
                'Eligible_Courses': list(eligible_courses - cumulative_courses)
            })

            # Determine Special Eligible Courses
            special_eligible_courses = {
                course for course in prerequisites_special.keys()
                if is_eligible_special(course, cumulative_courses, student_info, prerequisites_special, conditions)
            }
            final_results_special.append({
                'Student_ID': student_id,
                'Semester': semester,
                'Major': student_info['Major'],
                'College': student_info['College'],
                'Program': student_info['Program'],
                'Passed Credits': student_info['Passed Credits'],
                'Student_Level': student_info['Student_Level'],
                'Eligible_Courses': list(special_eligible_courses - cumulative_courses)
            })

    # Convert Results to DataFrames
    final_results_df = pd.DataFrame(final_results)
    final_results_special_df = pd.DataFrame(final_results_special)
    
    # Combine Eligible Courses from Both DataFrames
    combined_comp_list = combine_eligible_courses(final_results_df, final_results_special_df)
    # Find Course Combinations for Co-requisites
    combined_comp_list = combined_comp_list.apply(create_combined_courses, axis=1, co=courses_co)
    latest_eligible_courses = combined_comp_list.sort_values(by='Semester', ascending=False)
    latest_eligible_courses = latest_eligible_courses.groupby('Student_ID').first().reset_index()
    latest_eligible_courses = latest_eligible_courses.merge(grouped_data,on = "Student_ID",how = "inner")
    latest_eligible_courses["Eligible_Courses_CO"] = latest_eligible_courses.apply(remove_matches, axis=1)
    latest_eligible_courses = latest_eligible_courses.apply(process_row, axis=1)
    latest_eligible_courses.drop(columns=["Course_ID"], inplace=True)

    latest_eligible_courses = latest_eligible_courses.merge(merged_df, on='Student_ID', how='outer')
    latest_eligible_courses['Failed_Courses'] = latest_eligible_courses['Failed_Courses'].apply(lambda x: x if isinstance(x, list) else [])
    latest_eligible_courses['Eligible_Courses_CO'] = latest_eligible_courses['Eligible_Courses_CO'].apply(lambda x: x if isinstance(x, list) else [])
    latest_eligible_courses['Eligible_Courses_CO'] = latest_eligible_courses.apply(
        lambda row: list(set(row['Eligible_Courses_CO']) | (set(row['Failed_Courses']) - set(row['Eligible_Courses_CO']))),axis=1)
    latest_eligible_courses = latest_eligible_courses.drop(columns=['Failed_Courses'])

    latest_info_failed = failed_data.loc[failed_data.groupby("Student_ID")["Semester"].idxmax()]
    missing_semester_df = latest_eligible_courses[latest_eligible_courses['Semester'].isna()]
    latest_eligible_courses.dropna(inplace=True)
    columns_to_fill = ['Semester', 'Major', 'College', 'Program', 'Passed Credits', 'Student_Level']

    missing_semester_df = missing_semester_df.copy()
    
    for col in columns_to_fill:
        na_mask = missing_semester_df[col].isna()
        mapped_values = (
        missing_semester_df.loc[na_mask, 'Student_ID']
        .map(latest_info_failed.set_index('Student_ID')[col])
        )
        # Set full column values, replacing only where mask is True
        col_values = missing_semester_df[col].copy()
        col_values[na_mask] = mapped_values
        missing_semester_df[col] = col_values 

    columns_to_convert = ['Semester', 'Student_Level', 'Passed Credits']
    for col in columns_to_convert:
        latest_eligible_courses[col] = pd.to_numeric(latest_eligible_courses[col], errors='coerce').astype('Int64')
        
    latest_eligible_courses = pd.concat([latest_eligible_courses, missing_semester_df], ignore_index=True)

    max_semester_index = st_enrollment_data.groupby('Student_ID')['Semester'].idxmax()
    max_semester_data = st_enrollment_data.loc[max_semester_index, ['Student_ID', 'Semester']]

    last_semester_courses = pd.merge(max_semester_data, st_enrollment_data, on=['Student_ID', 'Semester'])
    eng097_fpu_students = last_semester_courses[last_semester_courses['Course_ID'] == 'ENGL097']
    # Target course list
    target_courses = ['ENGL098', 'MATH094', 'MATH095', 'MATH096', 'MATH098', 'MATH100', 'MATH111', 'MATH120', 'MATH121', 'MATH131', 'MATH140']

    eng097_fpu_students_eligible = latest_eligible_courses[latest_eligible_courses['Student_ID']
                                                       .isin(eng097_fpu_students['Student_ID'])].copy()
    eng097_fpu_students_eligible.loc[:, 'Eligible_Courses_CO'] = eng097_fpu_students_eligible['Eligible_Courses_CO'].apply(
    lambda courses: [course for course in courses if course in target_courses])

    latest_eligible_courses = latest_eligible_courses.merge(
    eng097_fpu_students_eligible[['Student_ID', 'Eligible_Courses_CO']],  # Relevant columns from filtered_students
    on='Student_ID',
    how='left',  # Keep all rows in students_df
    suffixes=('', '_updated'))  # Suffix to differentiate new column)

    latest_eligible_courses['Eligible_Courses_CO'] = latest_eligible_courses['Eligible_Courses_CO_updated'].combine_first(latest_eligible_courses['Eligible_Courses_CO'])
    latest_eligible_courses = latest_eligible_courses.drop(columns=['Eligible_Courses_CO_updated'])
    latest_eligible_courses = latest_eligible_courses.merge(grouped_data,on = "Student_ID",how = "outer")
    latest_eligible_courses['Course_ID'] = latest_eligible_courses['Course_ID'].apply(lambda x: x if isinstance(x, list) else [])
    latest_eligible_courses = latest_eligible_courses.apply(process_row, axis=1)
    latest_eligible_courses.drop(columns=["Course_ID"], inplace=True)

    # Exploding DataFrame and mapping course details
    eligible_courses_comprehensive_data = latest_eligible_courses.explode("Eligible_Courses_CO")
    eligible_courses_comprehensive_data = eligible_courses_comprehensive_data.merge(major_courses[['Course_ID', 'AREA_OF_STUDY', 'COURSE_OF_STUDY', 'Course_Level']],
                                                                                    left_on='Eligible_Courses_CO', right_on='Course_ID', how='left').drop(columns="Course_ID")
    eligible_courses_comprehensive_data['Eligible_Courses_CO'] = eligible_courses_comprehensive_data['Eligible_Courses_CO'].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

    # Find Additional Eligibilities
    eligible_courses_comprehensive_data['Future_Eligible_Courses_List'] = eligible_courses_comprehensive_data.apply(lambda row: find_additional_eligibilities(row['Eligible_Courses_CO'], set(row['Eligible_Courses_CO']), prerequisites), axis=1)
    eligible_courses_per_student = eligible_courses_comprehensive_data.groupby('Student_ID')['Eligible_Courses_CO'].agg(lambda x: list(set([item for sublist in x for item in sublist if isinstance(sublist, list)]))).reset_index()

    # Merge aggregated list back to the comprehensive DataFrame
    eligible_courses_comprehensive_data = eligible_courses_comprehensive_data.merge(eligible_courses_per_student.rename(columns={'Eligible_Courses_CO': 'Eligible_Courses_List_All'}), on='Student_ID', how='left')

    # Filter matching courses from future eligible lists
    eligible_courses_comprehensive_data['Future_Eligible_Courses_List'] = eligible_courses_comprehensive_data.apply(lambda row: [course for course in row['Future_Eligible_Courses_List'] if course not in row['Eligible_Courses_List_All']], axis=1)
    eligible_courses_comprehensive_data['Total_Future_Eligible_Courses_List'] = eligible_courses_comprehensive_data['Future_Eligible_Courses_List'].apply(len)

    # Special eligibility courses
    eligible_courses_comprehensive_data['Future_Eligible_Courses_Special'] = eligible_courses_comprehensive_data.apply(lambda row: find_additional_eligibilities_special(row['Eligible_Courses_CO'], set(row['Eligible_Courses_CO']), row, prerequisites_special, conditions, is_eligible_special_), axis=1)
    eligible_courses_comprehensive_data['Future_Eligible_Courses_Special'] = eligible_courses_comprehensive_data.apply(lambda row: [course for course in row['Future_Eligible_Courses_Special'] if course not in row['Eligible_Courses_List_All']], axis=1)
    eligible_courses_comprehensive_data['Total_Future_Eligible_Courses_Special'] = eligible_courses_comprehensive_data['Future_Eligible_Courses_Special'].apply(len)

    # Combine Future Eligible Courses and calculate the score
    eligible_courses_comprehensive_data["Future_Eligible_Courses"] = eligible_courses_comprehensive_data["Future_Eligible_Courses_List"] + eligible_courses_comprehensive_data["Future_Eligible_Courses_Special"]
    eligible_courses_comprehensive_data['Course_Score'] = eligible_courses_comprehensive_data['Future_Eligible_Courses'].apply(len)

    # Find Best Courses
    recommended_courses_comp = eligible_courses_comprehensive_data.groupby(['Student_ID', 'Semester']).apply(lambda group: pd.Series({'Recommended_Courses': find_best_courses(group)})).reset_index()


    eligible_courses_comprehensive_data = eligible_courses_comprehensive_data.merge(recommended_courses_comp, on=['Student_ID', 'Semester'])
    eligible_courses_comprehensive_data = eligible_courses_comprehensive_data.merge(weighted_remaining_courses_df, on=['Student_ID', 'AREA_OF_STUDY'], how='left')


    eligible_courses_comprehensive_data = eligible_courses_comprehensive_data.groupby('Student_ID', group_keys=False).apply(normalize_by_student)
    eligible_courses_comprehensive_data['Final_Score'] = (
        (eligible_courses_comprehensive_data['Normalized_Course_Score'] * 0.4) +
        (eligible_courses_comprehensive_data['Normalized_Remaining_Courses_Weight'] * 0.4) +
        (eligible_courses_comprehensive_data['Normalized_Course_Level'] * 0.2))

    # Find Best Courses
    recommended_courses_comp_v2 = eligible_courses_comprehensive_data.groupby(['Student_ID', 'Semester']).apply(lambda group: pd.Series({'Recommended_Courses_V2': find_best_courses_cea_v2(group)})).reset_index()
    eligible_courses_comprehensive_data = eligible_courses_comprehensive_data.merge(recommended_courses_comp_v2, on=['Student_ID', 'Semester'])

    recommended_courses = recommended_courses_comp.merge(recommended_courses_comp_v2,on=['Student_ID', 'Semester'])

    # Create summary DataFrames for eligible courses
    summary_area_of_study_eligible = eligible_courses_comprehensive_data.groupby(['Student_ID', 'AREA_OF_STUDY']).size().reset_index(name='Total_Eligible_Courses')
    summary_course_of_study_eligible = eligible_courses_comprehensive_data.groupby(['Student_ID', 'COURSE_OF_STUDY']).size().reset_index(name='Total_Eligible_Courses')
    summary_area_of_study_eligible = summary_area_of_study_eligible.pivot_table(index="Student_ID", columns="AREA_OF_STUDY", values="Total_Eligible_Courses", fill_value=0).reset_index()

    return requirements_,student_progress,summary_area_of_study_taken,remaining_courses_df,latest_eligible_courses,eligible_courses_comprehensive_data,recommended_courses,summary_area_of_study_eligible


# Provided mappings
college_groups = {
    "CBA": ['ACCOUNTING', 'INTL BUSIN', 'MANAGEMENT', 'FINANCE', 'MIS', 'MARKETING2'],
    "CAS": ['COMSCIENCE', 'ENGLISH', 'LINGUISTIC', 'LITERATURE', 'DIGITALMED', 'PR / ADV', 'VISUAL COM'],
    "CEA": ['COMPENG', 'ELECENG', 'MGMTENG']
}

major_mappings = {
    "Accounting": "ACCOUNTING",
    "International Business": "INTL BUSIN",
    "Mgmt & Organizational Behavior": "MANAGEMENT",
    "Computer Science": "COMSCIENCE",
    "Computer Engineering": "COMPENG",
    "Electrical Engineering": "ELECENG",
    "Engineering Management": "MGMTENG",
    "English Education": "ENGLISH",
    "Eng- Linguistics - Translation": "LINGUISTIC",
    "English Literature": "LITERATURE",
    "Finance": "FINANCE",
    "Digital Media Production": "DIGITALMED",
    "Public relations & Advertising": "PR / ADV",
    "Visual Communication": "VISUAL COM",
    "Management Information Systems": "MIS",
    "Marketing": "MARKETING2"
}

# Function to find the college group a major belongs to
def get_college_majors(major_code):
    for majors in college_groups.values():
        if major_code in majors:
            return majors
    return []

# Prepare a list with (major_name, major_code, related_college_majors)
loop_data = [
    (name, code, get_college_majors(code))
    for name, code in major_mappings.items()
]

# Displaying the generated list for confirmation
loop_data_df = pd.DataFrame(loop_data, columns=["Major Name", "Major Code", "College Majors"])

if navigation == "User Guide":
    st.title("User Guide")
    st.write("Welcome to the User Guide. Please choose an option below to learn more:")
    
    guide_option = st.selectbox("Choose an option:", ["Please select the required page!","Course Eligibility and Recommendation System", "Quick Check"])
    
    
    if guide_option == "Please select the required page!":
        st.info("No Page selected")

    if guide_option == "Course Eligibility and Recommendation System":
        with st.expander("Steps for Course Eligibility and Recommendation System"):
            st.markdown("""
                ### Select College and Major

                #### Choose a College:
                Use the dropdown labeled **"Select College"** to pick from the available colleges: **CBA**, **CAS**, or **CEA**.

                #### Choose Major(s):
                Once a college is selected, available majors will be displayed in a multi-select box. Select one or more majors to proceed.

                ### View Eligible and Recommended Courses

                #### Select Data Type:
                Use the **"Select Data to Display"** dropdown to choose the type of report you want:

                - **Major Sheet Requirements Data**: Shows course requirements for the selected major(s).
                - **Student Progress Report**: Displays the student's progress by area of study.
                - **Summary of Taken Courses by AREA_OF_STUDY**: Summarizes courses completed in each area of study.
                - **Remaining Courses by AREA_OF_STUDY**: Lists remaining courses needed.
                - **Latest Eligible Courses**: Shows the latest courses that the student is eligible to take.
                - **Comprehensive Eligible Courses Data**: Provides a comprehensive view of all eligible courses.
                - **Recommended Courses Report**: Recommends courses based on eligibility.
                - **Summary of Eligible Courses by AREA_OF_STUDY**: Summarizes eligible courses by study area.

                #### Viewing and Downloading Data

                ##### Viewing Data:
                Processed data will display in a table format.

                ##### Download Data as CSV:
                For each report, a download button is available to save the data as a **CSV file** for further use.
                """)

        
    elif guide_option == "Quick Check":
        with st.expander("Steps for Quick Check"):
            st.markdown("""
                ### Quick Check - Course Eligibility and Recommendation System

                #### 1. Input Semester Details
                You can specify multiple semesters, each with its own set of details:
                - **Number of Semesters to Add**: Use the number input to define how many semesters to include.
                - For each semester:
                - **Student ID**: Enter the unique ID for the student.
                - **Semester**: Specify the semester number.
                - **College**: Select the college from a dropdown (options: "CBA", "CAS", "CEA").
                - **Program**: Once a college is selected, choose the program (e.g., "Accounting" or "Finance").
                - **Major**: Based on the chosen program, select the specific major.
                - **Passed Credits**: Input the number of credits the student has already completed.
                - **Student Level**: Select the student’s current level (Freshman, Sophomore, Junior, Senior).
                - **Course ID**: Use the multi-select box to choose the courses from the loaded course list.
                - **Incoming PCR**: Enter the student's incoming PCR value.

                #### 2. Process Data
                - **Process Manual Input Data**: Check this box to begin processing the entered student data.

                #### 3. Select Data to Display
                After processing, choose from the following report options to view and download:

                - **Major Sheet Requirements Data**: Displays the course requirements for each major.
                - **Student Progress Report**: Shows student progress by area of study.
                - **Summary of Taken Courses by AREA_OF_STUDY**: Summarizes completed courses by study area.
                - **Remaining Courses by AREA_OF_STUDY**: Lists courses still required by study area.
                - **Latest Eligible Courses**: Shows the most recent courses the student is eligible for.
                - **Comprehensive Eligible Courses Data**: Provides a detailed view of all eligible courses.
                - **Recommended Courses Report**: Offers course recommendations based on eligibility.
                - **Summary of Eligible Courses by AREA_OF_STUDY**: Summarizes eligible courses in each study area.

                #### 4. Downloading Reports
                Each report is available for download in **CSV format**. Use the provided download button to save any displayed report as a CSV file.

                #### Note
                Ensure all required fields are correctly filled out before processing, or error messages will display.
                """)



elif navigation == "Course Eligibility and Recommendation System":
    st.title("Course Eligibility and Recommendation System")

    st.header("Select College & Major")
    selected_college = st.selectbox("Select College:", ["Please Select The Required College!"] + list(college_groups.keys()))

    if selected_college == "Please Select The Required College!":
        st.warning("No College Selected!")
        majors = []
    else:
        majors = [name for name, code in major_mappings.items() if code in college_groups[selected_college]]

    selected_major = st.multiselect("Select Major:", majors)

    st.header("Load Student Data")
    st.info("Please download the sample student data to understand the required format.")
    sample_data = create_sample_data()
    st.download_button(
        label="Download Sample Data",
        data=sample_data.to_csv(index=False),
        file_name='sample_student_data.csv',
        mime='text/csv',
    )
    student_file = st.file_uploader("Upload the Student Data ", type=["xlsx"])


    st.header("Eligible & Recommended Courses")

    if selected_college != "Please Select The Required College!" and selected_major:

        section = st.selectbox("Select Data to Display", [
            "None", "Major Sheet Requirements Data", "Student Progress Report", 
            "Summary of Taken Courses by AREA_OF_STUDY", "Remaining Courses by AREA_OF_STUDY", 
            "Latest Eligible Courses", "Comprehensive Eligible Courses Data", 
            "Recommended Courses Report", "Summary of Eligible Courses by AREA_OF_STUDY"
        ])

        if student_file:
            try:
                major_data = pd.read_excel("MajorSheet.xlsx", sheet_name=None)
                st_hist_data = pd.read_excel(student_file)
                #st_hist_data = st_data_cleaning(ac_st_enrollment_data, tc_data)
                #st_hist_data = st_hist_data[st_hist_data['Semester'] != 2501] 
            except Exception as e:
                st.error(f"Error loading sheets: {e}")
                st.stop()

            # Helper for downloads
            def download_df(df, filename):
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"Download {filename} as CSV",
                    data=csv,
                    file_name=filename,
                    mime='text/csv'
                )

            # Collect results across majors
            dataframes_list = []
            for major in selected_major:
                st.write(f"Processing data for major: {major}")
                major_code = major_mappings.get(major, "Unknown")
                major_data_subset = st_hist_data[st_hist_data['Major'] == major]
                college_majors = get_college_majors(major_code)

                try:
                    with st.spinner(f"Processing data for {major}..."):
                        result = process_data_generic(
                            major_data_subset,
                            major_data,
                            "Requierments_Weights.xlsx",
                            major,
                            major_code,
                            college_majors
                        )
                    dataframes_list.append(result)
                except Exception as e:
                    st.error(f"Error processing {major}: {e}")

            if dataframes_list:
                requirements = pd.concat([df[0] for df in dataframes_list], ignore_index=True)
                student_progress = pd.concat([df[1] for df in dataframes_list], ignore_index=True)
                summary_area_of_study_taken = pd.concat([df[2] for df in dataframes_list], ignore_index=False)
                remaining_courses = pd.concat([df[3] for df in dataframes_list], ignore_index=True)
                latest_eligible_courses = pd.concat([df[4] for df in dataframes_list], ignore_index=True)
                eligible_courses_comprehensive_data = pd.concat([df[5] for df in dataframes_list], ignore_index=True)
                recommended_courses = pd.concat([df[6] for df in dataframes_list], ignore_index=True)
                summary_area_of_study_eligible = pd.concat([df[7] for df in dataframes_list], ignore_index=True)

                st.success("Data processed successfully for all majors!")

                if section == "Major Sheet Requirements Data":
                    st.dataframe(requirements)
                    download_df(requirements, "requirements_df.csv")

                elif section == "Student Progress Report":
                    st.dataframe(student_progress)
                    download_df(student_progress, "student_progress.csv")

                elif section == "Summary of Taken Courses by AREA_OF_STUDY":
                    st.dataframe(summary_area_of_study_taken)
                    download_df(summary_area_of_study_taken, "summary_area_of_study_taken.csv")

                elif section == "Remaining Courses by AREA_OF_STUDY":
                    st.dataframe(remaining_courses)
                    download_df(remaining_courses, "remaining_courses_df.csv")

                elif section == "Latest Eligible Courses":
                    st.dataframe(latest_eligible_courses)
                    download_df(latest_eligible_courses, "latest_eligible_courses.csv")

                elif section == "Comprehensive Eligible Courses Data":
                    st.dataframe(eligible_courses_comprehensive_data)
                    download_df(eligible_courses_comprehensive_data, "eligible_courses_comprehensive_data.csv")

                elif section == "Recommended Courses Report":
                    st.dataframe(recommended_courses)
                    download_df(recommended_courses, "recommended_courses.csv")

                elif section == "Summary of Eligible Courses by AREA_OF_STUDY":
                    st.dataframe(summary_area_of_study_eligible)
                    download_df(summary_area_of_study_eligible, "summary_area_of_study_eligible.csv")

        else:
            st.warning("Please Choose the required Data!")

if navigation == "Quick Check":
    st.title("Course Eligibility and Recommendation System")

    num_semesters = st.number_input("Number of Semesters to Add:", min_value=1, value=1, step=1)
    student_info_list = []

    try:
        course_list = pd.read_excel("Course_ID.xlsx")['Course_ID'].tolist()
    except Exception as e:
        st.error(f"Error loading course list: {e}")
        course_list = []

    for i in range(num_semesters):
        st.subheader(f"Semester {i + 1} Information")

        student_id = st.text_input(f"Student ID (Semester {i + 1}):")
        semester = st.number_input(f"Semester (Semester {i + 1}):", min_value=1, value=1, step=1)
        college = st.selectbox(f"College (Semester {i + 1}):", ["CBA", "CAS", "CEA"], key=f"college_{i}")

        # Programs based on college groups
        college_programs = {
            "CBA": {
                "Accounting": ["Accounting"],
                "Finance": ["Finance"],
                "Marketing": ["Marketing"],
                "Management Information Systems": ["Management Information Systems"],
                "Business Administration": ["Mgmt & Organizational Behavior", "International Business"],
            },
            "CAS": {
                "Computer Science": ["Computer Science"],
                "English": ["English Education", "Eng- Linguistics - Translation", "English Literature"],
                "Mass Communication": ["Public relations & Advertising", "Digital Media Production", "Visual Communication"],
            },
            "CEA": {
                "Computer Engineering": ["Computer Engineering"],
                "Electrical Engineering": ["Electrical Engineering"],
                "Engineering Management": ["Engineering Management"],
            }
        }

        # Programs list by college
        college_programs_list = {
            "Please Select The Required College!": ["No College Selected!"],
            "CBA": ["Please Choose the required program!"] + list(college_programs["CBA"].keys()),
            "CAS": ["Please Choose the required program!"] + list(college_programs["CAS"].keys()),
            "CEA": ["Please Choose the required program!"] + list(college_programs["CEA"].keys()),
        }

        # --- Inside your loop ---
        if college == "Please Select The Required College!":
            st.warning("No College Selected!")

        programs = college_programs_list.get(college, ["No College Selected!"])
        program = st.selectbox(f"Program (Semester {i + 1}):", programs, key=f"program_{i}")

        if "Please Choose" in program or "No College" in program:
            st.warning("Please Choose the required program!")
            majors = ["No Program Selected!"]
        else:
            majors = college_programs.get(college, {}).get(program, ["No Program Selected!"])

        major = st.selectbox(f"Major (Semester {i + 1}):", majors, key=f"major_{i}")

        # Get internal major code from mappings
        major_code = major_mappings.get(major, "Unknown")
        passed_credits = st.number_input(f"Passed Credits (Semester {i + 1}):", value=0, min_value=0)
        student_level = st.selectbox(f"Student Level (Semester {i + 1}):", ["Freshman", "Sophomore", "Junior", "Senior"])
        course_id = st.multiselect(f"Course ID (Semester {i + 1}):", course_list, key=f"course_id_{i}")
        grades_list = ["P"]
        grades = st.multiselect(f"Grade (Semester {i + 1}):", grades_list)
        incoming_pcr = st.number_input(f"Incoming PCR (Semester {i + 1}):", value=0, min_value=0)

        student_info = {
            'Student_ID': student_id,
            'Semester': semester,
            'College': college,
            'Passed Credits': passed_credits,
            'Student_Level': {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4}[student_level],
            'Program': program,
            'Major': major,     # display name
            'Major_Code': major_code,  # internal code
            'Course_ID': course_id,
            'GRADE':grades,
            "Incoming_PCR": incoming_pcr
        }
        student_info_list.append(student_info)

    if st.checkbox("Process Manual Input Data"):
        try:
            major_data = pd.read_excel("MajorSheet.xlsx", sheet_name=None)
        except Exception as e:
            st.error(f"Error loading Major Sheet: {e}")
            major_data = {}

        if not student_info_list:
            st.error("Please fill in all required fields correctly before processing.")
        else:
            combined_data = pd.DataFrame(student_info_list).explode("Course_ID")
            combined_data["College"] = combined_data["College"].replace("CEA", "COE")

            st.success("Manual Data entered successfully!")
            st.table(combined_data)

            dataframes_list = []
            for _, row in combined_data.iterrows():
                college_majors = get_college_majors(row['Major_Code'])
                try:
                    result = process_data_generic(
                        combined_data[combined_data["Major"] == row['Major']],
                        major_data,
                        "Requierments_Weights.xlsx",
                        row['Major'],
                        row['Major_Code'],
                        college_majors
                    )
                    dataframes_list.append(result)
                except Exception as e:
                    st.error(f"Error processing {row['Major']}: {e}")

            if dataframes_list:
                requirements = pd.concat([df[0] for df in dataframes_list], ignore_index=True)
                student_progress = pd.concat([df[1] for df in dataframes_list], ignore_index=True)
                summary_area_of_study_taken = pd.concat([df[2] for df in dataframes_list], ignore_index=False)
                remaining_courses = pd.concat([df[3] for df in dataframes_list], ignore_index=True)
                latest_eligible_courses = pd.concat([df[4] for df in dataframes_list], ignore_index=True)
                eligible_courses_comprehensive_data = pd.concat([df[5] for df in dataframes_list], ignore_index=True)
                recommended_courses = pd.concat([df[6] for df in dataframes_list], ignore_index=True)
                summary_area_of_study_eligible = pd.concat([df[7] for df in dataframes_list], ignore_index=True)

                st.success("Data processed successfully!")
                section = st.selectbox("Select Data to Display", [
                    "None", "Major Sheet Requirements Data","Student Progress Report","Summary of Taken Courses by AREA_OF_STUDY",
                    "Remaining Courses by AREA_OF_STUDY","Latest Eligible Courses",
                    "Comprehensive Eligible Courses Data","Recommended Courses Report",
                    "Summary of Eligible Courses by AREA_OF_STUDY"
                ])

                def download_df(df, filename):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {filename} as CSV",
                        data=csv,
                        file_name=filename,
                        mime='text/csv'
                    )

                if section == "Major Sheet Requirements Data":
                    st.dataframe(requirements)
                    download_df(requirements, "requirements_df.csv")

                elif section == "Student Progress Report":
                    st.dataframe(student_progress)
                    download_df(student_progress, "student_progress.csv")

                elif section == "Summary of Taken Courses by AREA_OF_STUDY":
                    st.dataframe(summary_area_of_study_taken)
                    download_df(summary_area_of_study_taken, "summary_area_of_study_taken.csv")

                elif section == "Remaining Courses by AREA_OF_STUDY":
                    st.dataframe(remaining_courses)
                    download_df(remaining_courses, "remaining_courses_df.csv")

                elif section == "Latest Eligible Courses":
                    st.dataframe(latest_eligible_courses)
                    download_df(latest_eligible_courses, "latest_eligible_courses.csv")

                elif section == "Comprehensive Eligible Courses Data":
                    st.dataframe(eligible_courses_comprehensive_data)
                    download_df(eligible_courses_comprehensive_data, "eligible_courses_comprehensive_data.csv")

                elif section == "Recommended Courses Report":
                    st.dataframe(recommended_courses)
                    download_df(recommended_courses, "recommended_courses.csv")

                elif section == "Summary of Eligible Courses by AREA_OF_STUDY":
                    st.dataframe(summary_area_of_study_eligible)
                    download_df(summary_area_of_study_eligible, "summary_area_of_study_eligible.csv")
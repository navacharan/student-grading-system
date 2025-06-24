import streamlit as st
import random
import string
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize system in session_state to persist data
if 'sgs_system' not in st.session_state:
    class StudentGradingSystem:
        def __init__(self):
            self.admin_details = {"nicky": "135"}
            self.teacher_details = {"charan": "123"}
            self.student_data = {}
            self.courses = set()

        def generate_password(self, length=8):
            ch = string.ascii_letters + string.digits
            return ''.join(random.choice(ch) for _ in range(length))

        def verify_login(self, user_type, username, password):
            if user_type == "admin":
                return username in self.admin_details and self.admin_details[username] == password
            elif user_type == "teacher":
                return username in self.teacher_details and self.teacher_details[username] == password
            return False

        def add_student(self, sid, name, age):
            if sid in self.student_data:
                return "Student ID already exists."
            self.student_data[sid] = {"name": name, "age": age, "courses": {}, "grades": {}, "enrolled_courses": set()}
            return "Student added successfully!"

        def remove_student(self, sid):
            if sid not in self.student_data:
                return "Student not found!"
            del self.student_data[sid]
            return "Student removed successfully!"

        def add_teacher(self, tid, name, pwd):
            if name in self.teacher_details:
                return "Teacher name already exists."
            self.teacher_details[name] = pwd
            return "Teacher added successfully!"

        def remove_teacher(self, tid):
            if tid not in self.teacher_details:
                return "Teacher not found!"
            del self.teacher_details[tid]
            return "Teacher removed successfully!"

        def assign_course_to_student(self, teacher_id, sid, course):
            if sid not in self.student_data:
                return "Student not found!"
            self.student_data[sid]["courses"][course] = 0
            self.courses.add(course)
            return "Course assigned successfully!"

        def assign_grades(self, teacher_id, sid, course, grade):
            if sid not in self.student_data:
                return "Student not found!"
            if course not in self.student_data[sid]["courses"]:
                return "Course not found for student!"
            self.student_data[sid]["courses"][course] = grade
            self.student_data[sid]["grades"][course] = grade
            self.student_data[sid]["enrolled_courses"].add(course)
            return "Grade assigned successfully!"

        def view_teachers(self):
            return self.teacher_details

        def view_students(self):
            return self.student_data

        def view_student_report(self, sid):
            if sid not in self.student_data:
                return "Student not found!"
            return self.student_data[sid]["courses"]

        def search_student(self, sid):
            if sid not in self.student_data:
                return "Student not found!"
            return self.student_data[sid]

        def analyze_grades(self, course, passing_threshold=60):
            grades = []
            for student in self.student_data.values():
                if course in student["grades"]:
                    grades.append(student["grades"][course])

            if not grades:
                return "No grades found for this course."

            mean = statistics.mean(grades)
            median = statistics.median(grades)
            try:
                mode = statistics.mode(grades)
            except statistics.StatisticsError:
                mode = "No unique mode"
            stdev = statistics.stdev(grades)

            df = pd.DataFrame({'grade': grades, 'index': range(len(grades))})
            X = df[['index']]
            y = df['grade']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Linear Regression": LinearRegression(),
                "Polynomial Regression (degree 2)": PolynomialFeatures(degree=2),
                "Decision Tree Regression": DecisionTreeRegressor(),
                "Random Forest Regression": RandomForestRegressor(),
                "Logistic Regression (Binary)": LogisticRegression()
            }

            st.subheader(f"Grade Analysis for {course}")

            for name, model in models.items():
                try:
                    if name == "Polynomial Regression (degree 2)":
                        poly = model
                        X_poly = poly.fit_transform(X_train)
                        regressor = LinearRegression()
                        regressor.fit(X_poly, y_train)
                        predictions = regressor.predict(poly.transform(X))

                    elif name == "Logistic Regression (Binary)":
                        binary_y_train = [1 if grade >= passing_threshold else 0 for grade in y_train]
                        binary_y_test = [1 if grade >= passing_threshold else 0 for grade in y_test]

                        if len(set(binary_y_train)) < 2:
                            st.warning("Logistic Regression cannot be performed: only one class present.")
                            continue

                        model.fit(X_train, binary_y_train)
                        predictions = model.predict(X)
                        predictions_test = model.predict(X_test)

                        accuracy = accuracy_score(binary_y_test, predictions_test)
                        precision = precision_score(binary_y_test, predictions_test, zero_division=0)
                        recall = recall_score(binary_y_test, predictions_test, zero_division=0)
                        f1 = f1_score(binary_y_test, predictions_test, zero_division=0)

                        st.write(f"**Logistic Regression Metrics:**")
                        st.write(f"Accuracy: {accuracy:.2f}")
                        st.write(f"Precision: {precision:.2f}")
                        st.write(f"Recall: {recall:.2f}")
                        st.write(f"F1-score: {f1:.2f}")

                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X)

                    fig, ax = plt.subplots()
                    ax.scatter(df['index'], df['grade'], label='Actual Grades')
                    if name == "Logistic Regression (Binary)":
                        ax.plot(df['index'], [passing_threshold if p == 1 else 0 for p in predictions], label=f'{name} Prediction')
                    else:
                        ax.plot(df['index'], predictions, label=f'{name} Prediction')
                    ax.set_xlabel('Student Index')
                    ax.set_ylabel('Grades')
                    ax.set_title(f'{name} for {course}')
                    ax.legend()
                    st.pyplot(fig)

                except ValueError as e:
                    st.error(f"{name} Error: {str(e)}")

            return f"Mean: {mean}, Median: {median}, Mode: {mode}, Standard Deviation: {stdev}"

    st.session_state.sgs_system = StudentGradingSystem()

sgs_system = st.session_state.sgs_system

# Main App
def main():
    st.title("Student Grading System")

    user_type = st.selectbox("User Type", ["admin", "teacher"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if sgs_system.verify_login(user_type, username, password):
            st.success(f"Logged in as {user_type}")
            st.session_state.logged_in = True
            st.session_state.user_type = user_type
            st.session_state.user_id = username
        else:
            st.error("Login failed")

    if st.session_state.get('logged_in'):
        user_type = st.session_state['user_type']
        user_id = st.session_state['user_id']
        if user_type == "admin":
            admin_dashboard()
        elif user_type == "teacher":
            teacher_dashboard(user_id)

def admin_dashboard():
    st.header("Admin Dashboard")
    action = st.selectbox("Choose an action", [
        "Add Student", "Remove Student", "Add Teacher", "Remove Teacher",
        "View Teachers", "View Students", "View Student Report", "Search Student", "View All Courses"
    ])

    if action == "Add Student":
        sid = st.text_input("Student ID")
        name = st.text_input("Student Name")
        age = st.number_input("Student Age", min_value=0, step=1)
        if st.button("Add"):
            st.write(sgs_system.add_student(sid, name, age))

    elif action == "Remove Student":
        sid = st.text_input("Student ID")
        if st.button("Remove"):
            st.write(sgs_system.remove_student(sid))

    elif action == "Add Teacher":
        tid = st.text_input("Teacher ID")
        name = st.text_input("Teacher Name")
        pwd = st.text_input("Password")
        if st.button("Add"):
            st.write(sgs_system.add_teacher(tid, name, pwd))

    elif action == "Remove Teacher":
        tid = st.text_input("Teacher ID")
        if st.button("Remove"):
            st.write(sgs_system.remove_teacher(tid))

    elif action == "View Teachers":
        st.write("Teachers:")
        st.write(sgs_system.view_teachers())

    elif action == "View Students":
        st.write("Students:")
        st.dataframe(pd.DataFrame.from_dict(sgs_system.view_students(), orient="index"))

    elif action == "View Student Report":
        sid = st.text_input("Student ID")
        if st.button("View Report"):
            st.write(sgs_system.view_student_report(sid))

    elif action == "Search Student":
        sid = st.text_input("Student ID")
        if st.button("Search"):
            st.write(sgs_system.search_student(sid))

    elif action == "View All Courses":
        st.write("All Courses:")
        st.write(sgs_system.courses)

def teacher_dashboard(teacher_id):
    st.header("Teacher Dashboard")
    action = st.selectbox("Choose an action", [
        "Assign Course to Student", "Add Grade to Student", "View Students", "Analyze Grades"
    ])

    if action == "Assign Course to Student":
        sid = st.text_input("Student ID")
        course = st.text_input("Course Name")
        if st.button("Assign"):
            st.write(sgs_system.assign_course_to_student(teacher_id, sid, course))

    elif action == "Add Grade to Student":
        sid = st.text_input("Student ID")
        course = st.text_input("Course Name")
        grade = st.number_input("Grade", min_value=0.0, max_value=100.0)
        if st.button("Add Grade"):
            st.write(sgs_system.assign_grades(teacher_id, sid, course, grade))

    elif action == "View Students":
        st.write("Students:")
        st.dataframe(pd.DataFrame.from_dict(sgs_system.view_students(), orient="index"))

    elif action == "Analyze Grades":
        course = st.text_input("Course Name")
        passing_threshold = st.number_input("Passing Threshold", min_value=0, max_value=100, value=60)
        if st.button("Analyze"):
            result = sgs_system.analyze_grades(course, passing_threshold)
            st.write("Analysis Result:")
            st.write(result)

if __name__ == "__main__":
    main()

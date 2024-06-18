import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pickle
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt

class ChurniaMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Churnia Menu")
        self.root.configure(bg='#e01519')  # Set background color
        
        # Load the logo
        self.logo = tk.PhotoImage(file="./images/djezzy.png")

        self.menu_label = tk.Label(root, text="Predict a Client Churn", font=("Helvetica", 36, "bold"), fg="white", bg='#e01519')
        self.menu_label.pack(pady=40)

        self.manual_button = tk.Button(root, text="Manually", font=("Helvetica", 18), command=self.open_manual_prediction, bg="white", fg="#e81424", padx=20, pady=10, bd=10, relief="solid", highlightthickness=2, borderwidth=0, highlightbackground="#e81424")
        self.manual_button.pack(pady=10)


        self.file_button = tk.Button(root, text="From a File", font=("Helvetica", 18), command=self.open_from_file, bg="white", fg="#e81424", padx=20, pady=10, bd=10, relief="solid", highlightthickness=2, borderwidth=0, highlightbackground="#e81424")
        self.file_button.pack(pady=20)
        
        # Place the logo at the bottom
        self.logo_label = tk.Label(root, image=self.logo, bg='#e01519')
        self.logo_label.pack(side="bottom", pady=20)
        
        self.root.geometry("400x500")  # Set window width and height

    def open_manual_prediction(self):
        manual_root = tk.Toplevel()
        manual_root.title("Churnia - Manual Prediction")
        manual_root.configure(bg='#e01519')

        def store_values():
            selected_values["line_type"] = dropdown1_var.get()
            selected_values['devicetype'] = dropdown2_var.get()
            selected_values['value_segment'] = dropdown3_var.get()
            selected_values['global_profile'] = dropdown4_var.get()
            selected_values['age'] = float(numerical_var_age.get())
            selected_values['sex'] = dropdown6_var.get()
            selected_values['wilaya'] = dropdown7_var.get()
            selected_values['yr'] = float(numerical_var1.get())
            selected_values['mr'] = float(numerical_var2.get())
            selected_values['number_subscription'] = float(numerical_var3.get())
            selected_values['nb_supended'] = float(numerical_var4.get())
            return selected_values

        translation_mapping = {
            'wilaya': translate_wilaya,
            'yr': translate_yr,
            'mr': translate_mr,
            'nb_supended': translate_nb_supended,
            'number_subscription': translate_nbsubscription,
            'age': translate_age
        }

        def show_churn_risk():
            selected_values = store_values()
            translated_cust = {}

            for key, value in selected_values.items():
                if key in translation_mapping:
                    translation_function = translation_mapping[key]
                    translated_value = translation_function(value)
                    translated_cust[key] = translated_value
                else:
                    translated_cust[key] = value

            cust_df = pd.DataFrame(translated_cust, index=[0])
            desired_order = ['age', 'sex', 'wilaya', 'devicetype', 'line_type', 'global_profile', 'value_segment', 'number_subscription', 'nb_supended', 'yr', 'mr']

            cust_df = cust_df[desired_order]
            prediction = model.predict(cust_df)[0]

            if prediction == 0:
                conditional_label.config(text="Churn Risk: Low", font=("Helvetica", 36, "bold"), fg="white", bg='green')
            if prediction == 1:
                conditional_label.config(text="Churn Risk: High", font=("Helvetica", 36, "bold"), fg="white", bg='red')

        def store_and_predict():
            store_values()
            show_churn_risk()

        selected_values = {}

        rowl=0
        # Labels
        label5_text = "Age"
        label5_var = tk.StringVar(value=label5_text)  # Define StringVar for label 2
        label5 = tk.Label(manual_root, textvariable=label5_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label5.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label6_text = "Sex"
        label6_var = tk.StringVar(value=label6_text)  # Define StringVar for label 2
        label6 = tk.Label(manual_root, textvariable=label6_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label6.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label7_text = "Wilaya"
        label7_var = tk.StringVar(value=label7_text)  # Define StringVar for label 2
        label7 = tk.Label(manual_root, textvariable=label7_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label7.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label2_text = "Device Type"
        label2_var = tk.StringVar(value=label2_text)  # Define StringVar for label 2
        label2 = tk.Label(manual_root, textvariable=label2_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label2.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label1_text = "Line Type"
        label1_var = tk.StringVar(value=label1_text)  # Define StringVar for label 1
        label1 = tk.Label(manual_root, textvariable=label1_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label1.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label4_text = "Global Profile"
        label4_var = tk.StringVar(value=label4_text)  # Define StringVar for label 2
        label4 = tk.Label(manual_root, textvariable=label4_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label4.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label3_text = "Value Segment"
        label3_var = tk.StringVar(value=label3_text)  # Define StringVar for label 2
        label3 = tk.Label(manual_root, textvariable=label3_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label3.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label10_text = "Number of subscriptions"
        label10_var = tk.StringVar(value=label10_text)  # Define StringVar for label 2
        label10 = tk.Label(manual_root, textvariable=label10_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label10.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label11_text = "Number of suspensions"
        label11_var = tk.StringVar(value=label11_text)  # Define StringVar for label 2
        label11 = tk.Label(manual_root, textvariable=label11_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label11.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label8_text = "Yr"
        label8_var = tk.StringVar(value=label8_text)  # Define StringVar for label 2
        label8 = tk.Label(manual_root, textvariable=label8_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label8.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        label9_text = "Mr"
        label9_var = tk.StringVar(value=label9_text)  # Define StringVar for label 2
        label9 = tk.Label(manual_root, textvariable=label9_var, font=("Helvetica", 18, "bold"), fg="white", bg='#e01519', anchor='w')
        label9.grid(row=rowl, column=0, padx=10, pady=10, sticky="nsew")
        rowl+=1
        
        
        # Dropdowns
        options_line_type = ["2G", "3G", "4G"]
        options_device_type = ["Smartphone", "Phablet", "Feature Phone","Basic Phone","Router","Tablet",
                            "Mobile broadband PCI card","USB Modem","Weerable","IoT Device"]
        options_value_segment = ["NEW","Very High Value", "High Value","Medium", "Low Value","Very low Value",
                            "Zero Value", "Not concerned (postpayed)"
                            ]
        options_global_profile = ["POSTPAYED","PREPAYED", "Hybrid"]
        options_sex= ["Male","Female"]
        options_wilaya= [
            "ADRAR", "CHLEF", "LAGHOUAT", "OUM-EL-BOUAGHI", "BATNA", "BEJAIA", "BISKRA", 
            "BECHAR", "BLIDA", "BOUIRA", "TAMANRASSET", "TEBESSA", "TLEMCEN", "TIARET", 
            "TIZI-OUZOU", "ALGER", "DJELFA", "JIJEL", "SETIF", "SAIDA", "SKIKDA", 
            "SIDI-BELABBES", "ANNABA", "GUELMA", "CONSTANTINE", "MEDEA", "MOSTAGANEM", 
            "MSILA", "MASCARA", "OUARGLA", "ORAN", "EL-BAYADH", "ILLIZI", 
            "BORDJ-BOU-ARRERIDJ", "BOUMERDES", "EL-TARF", "TINDOUF", "TISSEMSILT", 
            "EL-OUED", "KHENCHELA", "SOUK-AHRAS", "TIPAZA", "MILA", "AIN-DEFLA", 
            "NAAMA", "AIN-TEMOUCHENT", "GHARDAIA", "RELIZANE", "TIMIMOUN", 
            "BORDJ-BADJI-MOKHTAR", "OULED-DJELLAL", "BENI-ABBES", "IN-SALAH", 
            "IN-GUEZZAM", "TOUGGOURT", "DJANET", "EL-MGHAIER", "EL-MENIAA"
        ]
        row =0
        numerical_var_age = tk.StringVar()  # StringVar for the numerical entry
        numerical_entry_age = tk.Entry(manual_root, textvariable=numerical_var_age, font=("Helvetica", 18) )
        numerical_entry_age.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        row+=1
        dropdown6_var = tk.StringVar()
        dropdown6 = ttk.Combobox(manual_root, textvariable=dropdown6_var, values=options_sex, font=("Helvetica", 18))
        dropdown6.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        dropdown6.current(0)
        row+=1
        dropdown7_var = tk.StringVar()
        dropdown7 = ttk.Combobox(manual_root, textvariable=dropdown7_var, values=options_wilaya, font=("Helvetica", 18))
        dropdown7.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        dropdown7.current(0)
        row+=1
        dropdown2_var = tk.StringVar()
        dropdown2 = ttk.Combobox(manual_root, textvariable=dropdown2_var, values=options_device_type, font=("Helvetica", 18))
        dropdown2.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        dropdown2.current(0)
        row+=1
        dropdown1_var = tk.StringVar()
        dropdown1 = ttk.Combobox(manual_root, textvariable=dropdown1_var, values=options_line_type, font=("Helvetica", 18))
        dropdown1.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        dropdown1.current(0)
        row+=1
        dropdown4_var = tk.StringVar()
        dropdown4 = ttk.Combobox(manual_root, textvariable=dropdown4_var, values=options_global_profile, font=("Helvetica", 18))
        dropdown4.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        dropdown4.current(0)
        row+=1
        dropdown3_var = tk.StringVar()
        dropdown3 = ttk.Combobox(manual_root, textvariable=dropdown3_var, values=options_value_segment, font=("Helvetica", 18))
        dropdown3.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        dropdown3.current(0)
        row+=1
        numerical_var3 = tk.StringVar()  # StringVar for the numerical entry
        numerical_entry3 = tk.Entry(manual_root, textvariable=numerical_var3, font=("Helvetica", 18))
        numerical_entry3.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        row+=1
        numerical_var4 = tk.StringVar()  # StringVar for the numerical entry
        numerical_entry4 = tk.Entry(manual_root, textvariable=numerical_var4, font=("Helvetica", 18))
        numerical_entry4.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        row+=1
        numerical_var1 = tk.StringVar()  # StringVar for the numerical entry
        numerical_entry1 = tk.Entry(manual_root, textvariable=numerical_var1, font=("Helvetica", 18))
        numerical_entry1.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        row+=1
        numerical_var2 = tk.StringVar()  # StringVar for the numerical entry
        numerical_entry2 = tk.Entry(manual_root, textvariable=numerical_var2, font=("Helvetica", 18))
        numerical_entry2.grid(row=row, column=1, padx=10, pady=10, sticky="nsew")
        row+=1
        # Button
        submit_button = tk.Button(manual_root, text="Predict", command=store_and_predict,
    font=("Helvetica", 14, "bold"),  # Set the font
    fg="red",  # Set the text color
    bg="white",  # Set the background color
    activeforeground="white",  # Set the text color when the button is clicked
    activebackground="red",  # Set the background color when the button is clicked
    borderwidth=0,  # Remove the border
    padx=20,  # Add horizontal padding
    pady=10  # Add vertical padding
    )
        submit_button.grid(row=row, columnspan=2, padx=10, pady=10, sticky="nsew")
        row+=1
        
        conditional_label = tk.Label(manual_root, text="", font=("Helvetica", 36, "bold"), fg="white", bg='#e01519')
        conditional_label.grid(row=row, columnspan=2, padx=10, pady=5, sticky="nsew")
        row += 1
        root.mainloop()

    def open_from_file(self):
        file_path = filedialog.askopenfilename(title="Select a data file")  # Open a file dialog to choose a file
        if file_path:
            self.process_file_data(file_path)  # Process the selected file's data

    def process_file_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data['age']=data['age'].apply(translate_age)
            data['wilaya']=data['wilaya'].apply(translate_wilaya)
            data['nb_supended']=data['nb_supended'].apply(translate_nb_supended)
            data['number_subscription']=data['number_subscription'].apply(translate_nbsubscription)
            data['yr']=data['yr'].apply(translate_yr)
            data['mr']=data['mr'].apply(translate_mr)
            desired_order = ['age', 'sex', 'wilaya', 'devicetype', 'line_type', 'global_profile', 'value_segment', 'number_subscription', 'nb_supended', 'yr', 'mr']
            data = data[desired_order]
            predictions = model.predict(data)  # Replace with your actual prediction result
            non_churned = (predictions == 0).sum()
            churned = (predictions == 1).sum()

            # Create a bar graph
            labels = ['Non-Churn', 'Churn']
            counts = [non_churned, churned]

            plt.bar(labels, counts)
            plt.xlabel('Churn')
            plt.ylabel('Count')
            plt.title('Churn vs Non-Churn')
            plt.show()
            
            messagebox.showinfo("Results", "There are {} non-churn and {} churn.".format(non_churned, churned))
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


root = tk.Tk()
app = ChurniaMenu(root)
model = pickle.load(open('churnModel.pkl', 'rb'))
def translate_wilaya(name):
    wilayas_code = {
                "ADRAR": 1,
                "CHLEF": 2,
                "LAGHOUAT": 3,
                "OUM-EL-BOUAGHI": 4,
                "BATNA": 5,
                "BEJAIA": 6,
                "BISKRA": 7,
                "BECHAR": 8,
                "BLIDA": 9,
                "BOUIRA": 10,
                "TAMANRASSET": 11,
                "TEBESSA": 12,
                "TLEMCEN": 13,
                "TIARET": 14,
                "TIZI-OUZOU": 15,
                "ALGER": 16,
                "DJELFA": 17,
                "JIJEL": 18,
                "SETIF": 19,
                "SAIDA": 20,
                "SKIKDA": 21,
                "SIDI-BELABBES": 22,
                "ANNABA": 23,
                "GUELMA": 24,
                "CONSTANTINE": 25,
                "MEDEA": 26,
                "MOSTAGANEM": 27,
                "MSILA": 28,
                "MASCARA": 29,
                "OUARGLA": 30,
                "ORAN": 31,
                "EL-BAYADH": 32,
                "ILLIZI": 33,
                "BORDJ-BOU-ARRERIDJ": 34,
                "BOUMERDES": 35,
                "EL-TARF": 36,
                "TINDOUF": 37,
                "TISSEMSILT": 38,
                "EL-OUED": 39,
                "KHENCHELA": 40,
                "SOUK-AHRAS": 41,
                "TIPAZA": 42,
                "MILA": 43,
                "AIN-DEFLA": 44,
                "NAAMA": 45,
                "AIN-TEMOUCHENT": 46,
                "GHARDAIA": 47,
                "RELIZANE": 48,
                "TIMIMOUN": 49,
                "BORDJ-BADJI-MOKHTAR": 50,
                "OULED-DJELLAL": 51,
                "BENI-ABBES": 52,
                "IN-SALAH": 53,
                "IN-GUEZZAM": 54,
                "TOUGGOURT": 55,
                "DJANET": 56,
                "EL-MGHAIER": 57,
                "EL-MENIAA": 58,
            }

    name_upper = name.upper()

    if name_upper in wilayas_code:
        return wilayas_code[name_upper]
    else:
        return "Error: Invalid wilaya name"   
            
def translate_age(age):
    if 14 <= age <= 19:
        return "Ado"
    elif 20 <= age <= 59:
        return "Adult"
    else:
        return "Senior"

def translate_yr(yr):
    yryr=(yr-0)/21
    return yryr

def translate_mr(mr):
    mrmr= (mr-0)/(11)
    return mrmr

def translate_nb_supended(nb_supended):
    nb_suspended_t = (nb_supended-0)/(9)
    return nb_suspended_t

def translate_nbsubscription(nbsubscription):
    nbsubscription_t= (nbsubscription-1)/(499)
    return nbsubscription_t

root.mainloop()
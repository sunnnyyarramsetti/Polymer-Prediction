import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

class SolubilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Solubility Prediction Tool")
        self.root.geometry("1000x750")
        self.root.resizable(True, True)

        # Background image setup
        try:
            bg_image = Image.open("15000.jpg").resize((1000, 750), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.bg_label = tk.Label(root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            self.bg_label = tk.Label(root, bg="#ecf0f1")
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load data
        self.df_exp = pd.read_csv(r"C:\Users\Jaya Sai Sunny\Downloads\experimental_dataset.csv")
        self.df_pinfo = pd.read_csv(r"C:\Users\Jaya Sai Sunny\Downloads\list_of_polymers.csv")
        self.df_sinfo = pd.read_csv(r"C:\Users\Jaya Sai Sunny\Downloads\list_of_solvents.csv")

        self.model = None
        self.df_clean = None
        self.filtered_df = None

        self.build_ui()

    def build_ui(self):
        main_frame = tk.Frame(self.root, bg="#ecf0f1", bd=2, relief=tk.RIDGE)
        main_frame.place(relx=0.5, rely=0.5, anchor="center", width=900, height=650)

        tk.Label(main_frame, text="Solubility Prediction Tool", font=("Helvetica", 20, "bold"),
                 bg="#2980b9", fg="white", pady=10).pack(fill=tk.X)

        form_frame = tk.Frame(main_frame, bg="#ecf0f1")
        form_frame.pack(pady=15)

        tk.Label(form_frame, text="Select Polymer:", font=("Helvetica", 12), bg="#ecf0f1").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.polymer_var = tk.StringVar()
        tk.OptionMenu(form_frame, self.polymer_var, *self.df_pinfo['name'].dropna().unique()).grid(row=0, column=1, padx=10, pady=10, sticky="w")

        tk.Label(form_frame, text="Select Solvent:", font=("Helvetica", 12), bg="#ecf0f1").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.solvent_var = tk.StringVar()
        tk.OptionMenu(form_frame, self.solvent_var, *self.df_sinfo['name'].dropna().unique()).grid(row=1, column=1, padx=10, pady=10, sticky="w")

        btn_frame = tk.Frame(main_frame, bg="#ecf0f1")
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Run Model", font=("Helvetica", 12, "bold"), bg="#3498db", fg="white",
                  command=self.run_model).pack(side="left", padx=10)

        tk.Button(btn_frame, text="Show Visualizations", font=("Helvetica", 12, "bold"), bg="#2ecc71", fg="white",
                  command=self.show_visualizations).pack(side="left", padx=10)

        out_frame = tk.Frame(main_frame, bg="#ecf0f1")
        out_frame.pack(pady=15)

        tk.Label(out_frame, text="\U0001F4E5 Input Details", font=("Helvetica", 12, "bold"),
                 bg="#2980b9", fg="white", width=40).grid(row=0, column=0, padx=10)
        tk.Label(out_frame, text="\U0001F4CA Prediction Results", font=("Helvetica", 12, "bold"),
                 bg="#2980b9", fg="white", width=40).grid(row=0, column=1, padx=10)

        self.input_text = tk.Text(out_frame, height=6, width=50, font=("Courier", 10), bg="#d6eaf8")
        self.input_text.grid(row=1, column=0, padx=10, pady=5)

        self.output_text = tk.Text(out_frame, height=6, width=50, font=("Courier", 10), bg="#d6eaf8")
        self.output_text.grid(row=1, column=1, padx=10, pady=5)

        self.plot_frame = tk.Frame(main_frame, bg="#ffffff", bd=1, relief=tk.SUNKEN)
        self.plot_frame.pack(pady=10, fill="both", expand=True)

    def run_model(self):
        try:
            polymer_name = self.polymer_var.get()
            solvent_name = self.solvent_var.get()

            if not polymer_name or not solvent_name:
                messagebox.showwarning("Missing Selection", "Please select both polymer and solvent.")
                return

            polymer_id_series = self.df_pinfo[self.df_pinfo['name'].str.lower().str.strip() == polymer_name.lower().strip()]['polymer']
            solvent_id_series = self.df_sinfo[self.df_sinfo['name'].str.lower().str.strip() == solvent_name.lower().strip()]['solvent']

            if polymer_id_series.empty or solvent_id_series.empty:
                messagebox.showwarning("Invalid Selection", "Selected polymer or solvent not found in the dataset.")
                return

            polymer_id = polymer_id_series.values[0]
            solvent_id = solvent_id_series.values[0]

            df_merge = pd.merge(self.df_pinfo, self.df_exp, on="polymer", how="left")
            df_n = pd.merge(self.df_sinfo, df_merge, on="solvent", how="left")
            df_n["sol-pol"] = df_n["solvent"].astype(str) + "-" + df_n["polymer"].astype(str)
            self.filtered_df = df_n[(df_n["polymer"] == polymer_id) & (df_n["solvent"] == solvent_id)].dropna()

            if self.filtered_df.empty:
                messagebox.showwarning("Combination Not Found", "No data found for the selected polymer-solvent combination.")
                return

            features = ["temperature", "pressure", "dens", "tg", "mw", "mn", "cryst"]
            self.df_clean = self.filtered_df[features + ["wa", "sol-pol"]].dropna()
            X = self.df_clean[features]
            y = self.df_clean["wa"]

            if X.empty or y.empty:
                messagebox.showwarning("Insufficient Data", "The dataset for this combination lacks sufficient feature or label data.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            self.model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=-1, verbosity=-1)
            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30)])

            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, f"Selected Polymer: {polymer_name}\nSelected Solvent: {solvent_name}")

            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"RMSE: {rmse:.4f}\nR\u00b2 Score: {r2:.4f}")

            self.plot_feature_importance()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def plot_feature_importance(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        importances = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values(by="importance", ascending=True)

        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='importance', y='feature', data=importances, ax=ax, palette="Blues")
        ax.set_title("Feature Importance")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_visualizations(self):
        if self.filtered_df is None or self.model is None:
            messagebox.showwarning("Run Model First", "Please run the model first to generate visualizations.")
            return

        features = ["temperature", "pressure", "dens", "tg", "mw", "mn", "cryst"]
        
        # Create visualization window
        vis_window = tk.Toplevel(self.root)
        vis_window.title("All Visualizations")
        vis_window.state('zoomed')  # Maximize window
        
        # Create main container
        main_frame = tk.Frame(vis_window)
        main_frame.pack(fill="both", expand=True)
        
        # Create canvas and scrollbar with smooth scrolling
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        # Use a frame to hold all visualizations
        vis_container = tk.Frame(canvas)
        
        # Configure canvas scrolling
        vis_container.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all"),
                width=e.width  # Fix width to prevent blurring
            )
        )
        
        # Create window in canvas with center anchor
        canvas.create_window((canvas.winfo_width()//2, 0), 
                           window=vis_container, 
                           anchor="n",
                           width=canvas.winfo_width())
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Set DPI for high-quality plots
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        
        # Feature Importance plot
        fig_imp = plt.figure(figsize=(10, 5), dpi=100)
        importances = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values(by="importance", ascending=True)
        
        sns.barplot(x='importance', y='feature', data=importances, palette="mako")
        plt.title("Feature Importances")
        plt.tight_layout()
        
        # Create frame for this plot
        frame_imp = tk.Frame(vis_container)
        frame_imp.pack(pady=10, fill="x")
        
        canvas_imp = FigureCanvasTkAgg(fig_imp, master=frame_imp)
        canvas_imp.draw()
        canvas_imp.get_tk_widget().pack(fill="x")
        
        # Boxplots for each feature
        for feature in features:
            fig_box = plt.figure(figsize=(10, 5), dpi=100)
            sns.boxplot(data=self.filtered_df, x=feature, y='wa')
            plt.title(f'Solubility vs {feature.capitalize()}')
            plt.tight_layout()
            
            # Create frame for each plot
            frame_box = tk.Frame(vis_container)
            frame_box.pack(pady=10, fill="x")
            
            canvas_box = FigureCanvasTkAgg(fig_box, master=frame_box)
            canvas_box.draw()
            canvas_box.get_tk_widget().pack(fill="x")
        
        # Bind canvas resize to keep content centered
        def on_canvas_resize(event):
            canvas.itemconfig(1, width=event.width)  # Update window width
            canvas.coords(1, (event.width//2, 0))   # Keep centered
            
        canvas.bind("<Configure>", on_canvas_resize)
        
        vis_window.lift()
        vis_window.focus_force()

if __name__ == '__main__':
    root = tk.Tk()
    try:
        root.state('zoomed')
    except:
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.update_idletasks()
    app = SolubilityApp(root)
    root.mainloop()

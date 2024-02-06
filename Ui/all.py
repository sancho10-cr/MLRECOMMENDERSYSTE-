import tkinter as tk
from tkinter import ttk
import os
import subprocess

def launch_movie_ui():
    subprocess.Popen(["python", "D:\\codes\\python\\Recomendation\\movie recomender\\movieUI.py"])

def launch_tweet_ui():
    subprocess.Popen(["python", "D:\\codes\\python\\Recomendation\\movie recomender\\TwitterUi.py"])

# Create the main window
root = tk.Tk()
root.title("Recommender System Selector")

# Label to prompt the user to select a recommender system
label = tk.Label(root, text="Select a Recommender System:")
label.pack()

# Dropdown menu for selecting the recommender system
selected_system = tk.StringVar()
system_menu = ttk.Combobox(root, textvariable=selected_system, values=["Movie Recommender", "Tweet Recommender"])
system_menu.pack()

# Button to launch the selected recommender system UI
launch_button = tk.Button(root, text="Launch", command=lambda: launch_movie_ui() if selected_system.get() == "Movie Recommender" else launch_tweet_ui())
launch_button.pack()

# Start the main loop
root.mainloop()

import tkinter as tk
from tkinter import ttk
import os

def print_text(entry, variable_name):
    # Retrieve the text from the text box
    user_text = entry.get()
    # Print the text to the console
    print(f'{variable_name}: {user_text}')


# Create the main application window
root = tk.Tk()
root.title("Multi Input Text Printer")

# Dictionary to hold input labels and their corresponding variable names
input_fields = {
    "Epoch": "epoch",
    "Batch Size": "batch_size",
    "Learning Rate": "learning_rate",
    "Voices": "voices",
    "Model Name": "model_name"
}

# List to hold input entries
entries = []

for label, variable in input_fields.items():
    # Create a label for each text entry
    text_label = tk.Label(root, text=f"Enter your text for {label}:")
    text_label.pack(pady=5)

    # Create the text entry widget
    text_entry = tk.Entry(root, width=40)
    text_entry.pack(pady=5)

    # Create a button that prints the text to the console
    print_button = tk.Button(root, text=f"Print {label}",
                             command=lambda e=text_entry, v=variable: print_text(e, v))
    print_button.pack(pady=5)

    # Store entry widget in the list
    entries.append(text_entry)

# start training
def start_training():
    print("start training")
# Create a button that prints the text to the console
training_btn = tk.Button(root, text=f"Start training",
                         command=start_training)
training_btn.pack(pady=5)

# start validation
def start_validation():
    print("start validation")
# Create a button that prints the text to the console
validation_btn = tk.Button(root, text=f"Start validation",
                         command=start_validation)
validation_btn.pack(pady=5)

# Create a label for the dropdown menu
dropdown_label = tk.Label(root, text="Select a Model:")
dropdown_label.pack(pady=10)

# Define the folder path
folder_path_models = '../code/models'  # This refers to the project root directory

# Get the list of files and directories
all_models = os.listdir(folder_path_models)

# List of variables for the dropdown menu
variables_list = ["car", "mouse", "house"]

# Create the dropdown menu (combobox)
selected_variable = tk.StringVar()
dropdown = ttk.Combobox(root, textvariable=selected_variable, state="readonly")

dropdown['values'] = all_models
dropdown.current(0)  # Set the default selection to the first item
dropdown.pack(pady=10)

def print_selected_variable():
    print(f'Selected variable: {selected_variable.get()}')

# Create a button to print the selected variable
dropdown_button = tk.Button(root, text="Print Selected Variable", command=print_selected_variable)
dropdown_button.pack(pady=10)

#??????????????

# Create a label for the dropdown menu
dropdown_label2 = tk.Label(root, text="Select a Save Model:")
dropdown_label2.pack(pady=10)

# Define the folder path
folder_path = '../generated_models'  # This refers to the project root directory

# Get the list of files and directories
all_files = os.listdir(folder_path)

print(all_files)
# List of variables for the dropdown menu
model_list = ["v1", "v2", "v3"]

# Create the dropdown menu (combobox)
selected_model = tk.StringVar()
dropdown2 = ttk.Combobox(root, textvariable=selected_model, state="readonly")

dropdown2['values'] = all_files
dropdown2.current(0)  # Set the default selection to the first item
dropdown2.pack(pady=10)


# Function to print the selected dropdown value
def print_selected_model():
    print(f'Selected variable: {selected_model.get()}')


# Create a button to print the selected variable
dropdown_button2 = tk.Button(root, text="Print Selected Version", command=print_selected_model)
dropdown_button2.pack(pady=10)


###########################

def add_item():
    text = entry.get()
    if text:
        frame = tk.Frame(items_frame)
        frame.pack(fill=tk.X, pady=2)

        label = tk.Label(frame, text=text, anchor='w')
        label.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        delete_button = tk.Button(frame, text="x", command=lambda f=frame: delete_item(f))
        delete_button.pack(side=tk.LEFT)

        items.append(frame)
        entry.delete(0, tk.END)
        update_scrollregion()

def delete_item(frame):
    frame.destroy()
    items.remove(frame)
    update_scrollregion()

def update_scrollregion():
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox('all'))

def on_canvas_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))


text_label = tk.Label(root, text=f"Enter your words:")
text_label.pack(pady=5)

entry = tk.Entry(root, width=40)
entry.pack(pady=10)

add_button = tk.Button(root, text="Add", command=add_item)
add_button.pack(pady=5)

# Create a frame for the canvas and scrollbar
list_frame = tk.Frame(root)
list_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# Create a canvas with a specified height
canvas = tk.Canvas(list_frame, height=200)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the canvas
scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas to use the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', on_canvas_configure)

# Create an internal frame to hold the list items
items_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=items_frame, anchor='nw')

items = []

# Start the Tkinter event loop
root.mainloop()
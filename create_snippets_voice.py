import yt_dlp
import moviepy.editor as mp
import os
import tkinter as tk


def create_snippet(link,name, timestamp, amount=1):

    url = link

    ydl_opts = {
        'format': 'wav/bestaudio/best',
        'outtmpl': 'mytitle',
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)
    except yt_dlp.DownloadError as e:
        print(e)
        return

    print("done")
    path = f"C:/Users/blura/Desktop/Voice Bachelor proj/voices/{name}/"
    if not os.path.exists(path):
        os.umask(0)
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")
    # 1 sec
    length = 10
    if not amount or amount <= 0:
        amount = 1

    print(f"timestamp {timestamp} amount {amount}")
    for i in range(0, amount):
        audio = mp.AudioFileClip("../mytitle.wav")
        # Generate a random starting point for the snippet
        start_time = int(timestamp) + (i * length)

        # Create a 10-second snippet starting from the random position
        snippet = audio.subclip(start_time, start_time + length)

        # Export the snippet to a new file
        i = 0
        audio_file_name = f"C:/Users/blura/Desktop/Voice Bachelor proj/voices/{name}/snippet{i}.wav"
        while os.path.exists(audio_file_name):
            i += 1
            audio_file_name = f"C:/Users/blura/Desktop/Voice Bachelor proj/voices/{name}/snippet{i}.wav"

        snippet.write_audiofile(audio_file_name)
        snippet.close()


def store_values_enter(event):
    link = entry_string.get()
    timestamp = int(entry_number.get())
    amount = int(entry_amount.get())
    name = entry_name.get()

    # Clear the inputs
    entry_string.delete(0, tk.END)
    entry_number.delete(0, tk.END)
    entry_amount.delete(0, tk.END)
    entry_name.delete(0, tk.END)

    # Call your function (replace this with your actual function)
    print(f"link {link}, timestamp: {timestamp} amount: {amount}")
    create_snippet(link,name, timestamp,amount)


def store_values():
    link = entry_string.get()
    timestamp = int(entry_number.get())
    amount = int(entry_amount.get())
    name = entry_name.get()

    # Clear the inputs
    entry_string.delete(0, tk.END)
    entry_number.delete(0, tk.END)
    entry_amount.delete(0, tk.END)
    entry_name.delete(0, tk.END)

    # Call your function (replace this with your actual function)
    print(f"link {link}, timestamp: {timestamp} amount {amount}")
    create_snippet(link,name, timestamp, amount)


# Create the main window
window = tk.Tk()
window.title("Data Input GUI")
window.geometry("500x200")

# Create StringVar and IntVar to store the input values
entry_string_var = tk.StringVar()
entry_number_var = tk.StringVar()
entry_name_var = tk.StringVar()
entry_amount_var = tk.StringVar(value="4")

# Create entry widgets
entry_string = tk.Entry(window, textvariable=entry_string_var, width=50)
entry_number = tk.Entry(window, textvariable=entry_number_var, width=50)
entry_name = tk.Entry(window, textvariable=entry_name_var, width=50)
entry_amount = tk.Entry(window, textvariable=entry_amount_var, width=50)

# Create labels
label_string = tk.Label(window, text="Enter String:")
label_number = tk.Label(window, text="Enter timestamp:")
label_amount = tk.Label(window, text="Enter amount:")
label_name = tk.Label(window, text="Enter name:")

# Create button
enter_button = tk.Button(window, text="Enter", command=store_values)

# Grid layout with increased width
label_string.grid(row=0, column=0, padx=10, pady=10)
entry_string.grid(row=0, column=1, padx=10, pady=10, columnspan=2)
label_number.grid(row=1, column=0, padx=10, pady=10)
entry_number.grid(row=1, column=1, padx=10, pady=10, columnspan=2)
label_name.grid(row=2, column=0, padx=10, pady=10)
entry_name.grid(row=2, column=1, padx=10, pady=10, columnspan=2)
label_amount.grid(row=3, column=0, padx=10, pady=10)
entry_amount.grid(row=3, column=1, padx=10, pady=10, columnspan=2)
enter_button.grid(row=4, column=0, columnspan=3, pady=10)

window.bind('<Return>', store_values_enter)

# Run the main loop
window.mainloop()

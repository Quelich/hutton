# import tkinter as tk
# from tkinter import filedialog, NW, SW, S
# from PIL import ImageTk, Image


# class Hutton_App(tk.Frame):
#     def __init__(self, master=None):
#         super().__init__(master)
#         self.img_instance = None
#         self.master = master
#         self.pack()
#         self.create_widgets()
#         self.master.geometry("960x720")
#         self.img_instance = None
#         self.img_file_path = None

#     def create_widgets(self):
#         # Creates a Select Button that lets the user to open a file directory
#         self.select_button = tk.Button(self)
#         self.select_button["text"] = "Select Image\n(click me)"
#         self.select_button["command"] = self.get_dir
#         self.select_button.pack(side="top")

#         # Retrieving an image file from the user
#         self.canvas = tk.Canvas(root, width=300, height=300)
#         self.canvas.pack(side="bottom")
#         img_file_path = self.get_dir()

#         # Displaying the selected image
#         self.img_instance = self.store_image(img_file_path)
#         self.display_image()

#         # Quit widget of the app
#         self.quit = tk.Button(self, text="QUIT", fg="red",
#                               command=self.master.destroy)
#         self.quit.pack(side="bottom")

#     def get_dir(self):
#         self.img_file_path = filedialog.askopenfilename(initialdir="/", title="Select file")
#         print(self.img_file_path)
#         return self.img_file_path

#     def store_image(self, img_file_path):
#         open_img = Image.open(img_file_path)
#         img_instance = ImageTk.PhotoImage(open_img)
#         return img_instance

#     def display_image(self):
#         tk.Label(root, text='Position image on button',
#                  font=('Consolas', 16)).pack(side="bottom", padx=480, pady=360)
#         self.canvas.create_image(200, 200, image=self.img_instance, anchor=tk.S)
#         self.canvas.image = self.img_instance
        
#     def make_entry(self):
#         entry = tk.Entry(self, width=20)
#         return entry


# root = tk.Tk()
# hutton_app = Hutton_App(master=root)
# hutton_app.master.title("Hutton: Rock Classifier")
# hutton_app.mainloop()

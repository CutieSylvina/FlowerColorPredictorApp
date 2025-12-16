import tkinter as tk
from PIL import Image, ImageTk
from color_pred.color_predict import ColorPredict

class MainApp(tk.Tk): 
    def __init__(self): 
        super().__init__()
        self.title("Flower Color Predictor App")
        
        # Get screen width and height
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        
        # Set window size to match screen 
        self.geometry(f"{self.screen_width}x{self.screen_height}")
       
        # Make it fullscreen 
        self.attributes("-fullscreen", True)
        
        # Load and resize background image to fit the screen 
        self.bg_image = Image.open("./resources/bg.jpg")
        self.bg_image = self.bg_image.resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.bg = ImageTk.PhotoImage(self.bg_image)
        
        self.header_image=Image.open("./resources/header.png")
        header_height = int(self.screen_height * 0.20)
        header_width = int(self.screen_width)

        self.header_image = self.header_image.resize((header_width, header_height),Image.LANCZOS)
        self.header_img = ImageTk.PhotoImage(self.header_image)
        
        # Place background image to fill the entire window
        self.background = tk.Label(self, image=self.bg)
        self.background.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.header = tk.Canvas(self, width=self.screen_width, height=((self.screen_height)*0.20), highlightthickness=0) 
        self.header.pack()
        
        self.header.create_image(0, 0, anchor="nw", image=self.header_img)
        
        
        # Place buttons 
        
        self.button0 = tk.Button(self, text="Predict Color", font=("Arial", "18", "bold"),  width=20, bg="#D79BC5", highlightbackground="#76121E", command=self.open_color_predict_screen)
        self.button0.place(anchor="center", relx=0.5, rely=0.45)
        
        # self.button1 = tk.Button(self, text="Dominant Colors", font=("Arial", "18", "bold"),  width=20, bg="#D79BC5", highlightbackground="#76121E",  command=self.open_color_identifier_screen)
        # self.button1.place(anchor="center", relx=0.5, rely=0.45)
        
        # self.button2 = tk.Button(self, text="About", width=20, font=("Arial","18", "bold"),bg="#D79BC5", highlightbackground="#76121E")
        # self.button2.place(anchor="center", relx=0.5, rely=0.55)
        
        self.button3 = tk.Button(self, text="Exit", font=("Arial", "18", "bold"), width= 20,bg="#D79BC5", highlightbackground="#76121E",command=self.destroy)
        self.button3.place(anchor="center", relx=0.5, rely=0.55)
         
        self.footer_image=Image.open("./resources/footer.png")
        footer_height = int(self.screen_height * 0.20)
        footer_width = int(self.screen_width)

        self.footer_image = self.footer_image.resize((footer_width, footer_height),Image.LANCZOS)
        self.footer_img = ImageTk.PhotoImage(self.footer_image)
          
        # Footer  
        self.footer = tk.Canvas(self, width=self.screen_width, height=((self.screen_height)*0.10), bg="#792842", highlightthickness=0) 
        self.footer.pack(side="bottom")
        
        self.footer.create_image(0, 0, anchor="nw", image=self.footer_img)
        
    # def open_color_identifier_screen(self): 
    #     self.withdraw()
    #     ColorIdentifier(self)  
    
    def open_color_predict_screen(self): 
        self.withdraw()
        ColorPredict(self)
        
MainApp().mainloop() 
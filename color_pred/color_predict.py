import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import cv2
from ml_algo.kmeans import KMeansCustom
from ml_algo.knn import KNNSimple
from ml_algo.randomForest import RandomForestClassifierSimple

class ColorPredict(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Color Predict")
        self.rhs_colors = []
        self.clicked_pixels = []

        self.load_rhs()
        self.train_models()

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(f"{self.screen_width}x{self.screen_height}")
         # Make it fullscreen 
        self.attributes("-fullscreen", True)

        self.background = tk.Label(self, bg="#D14D72")
        self.background.place(x=0, y=0, relwidth=1, relheight=1)

        # Buttons
        self.upload_button = tk.Button(self, text="Upload an Image", width=30, font=("Arial",12,"bold"),
                                       bg="#FCC8D1", command=self.upload_image)
        self.upload_button.pack(padx=20, pady=5)
        self.return_button = tk.Button(self, text="Return", font=("Arial",12,"bold"),
                                       bg="#FCC8D1", command=self.go_back)
        self.return_button.pack(pady=5)
        self.report_button = tk.Button(self, text="Generate Accuracy Report", font=("Arial",12,"bold"),
                                       bg="#FCC8D1", command=self.generate_report)
        self.report_button.pack(pady=5)

        # Grid layout
        grid_frame = tk.Frame(self, bg="#D14D72")
        grid_frame.pack(expand=True, fill="both")
        for i in range(2): grid_frame.grid_columnconfigure(i, weight=1)
        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_rowconfigure(1, weight=2)

        # Original Image
        self.top_left = tk.Frame(grid_frame, bg="#FEF2F4")
        self.top_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tk.Label(self.top_left, text="Original Image", bg="#FEF2F4", font=("Arial",12,"bold")).pack(pady=5)
        self.top_left_canvas = tk.Canvas(self.top_left, bg="#FEF2F4")
        self.top_left_canvas.pack(expand=True, fill="both")

        # Corrected Image
        self.top_right = tk.Frame(grid_frame, bg="#FEF2F4")
        self.top_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tk.Label(self.top_right, text="Corrected Image", bg="#FEF2F4", font=("Arial",12,"bold")).pack(pady=5)
        self.top_right_canvas = tk.Canvas(self.top_right, bg="#FEF2F4")
        self.top_right_canvas.pack(expand=True, fill="both")
        self.top_right_canvas.bind("<Button-1>", self.get_pixel_info)

        # Bottom scrollable panel
        self.bottom = tk.Frame(grid_frame, bg="#FEF2F4")
        self.bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        bottom_container = tk.Frame(self.bottom)
        bottom_container.pack(fill="both", expand=True)
        canvas_frame = tk.Frame(bottom_container)
        canvas_frame.pack(fill="both", expand=True)
        self.bottom_canvas = tk.Canvas(canvas_frame)
        self.bottom_canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.bottom_canvas.yview)
        self.v_scroll.pack(side="right", fill="y")
        self.h_scroll = tk.Scrollbar(bottom_container, orient="horizontal", command=self.bottom_canvas.xview)
        self.h_scroll.pack(side="bottom", fill="x") 
        self.bottom_canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )
        self.bottom_inner = tk.Frame(self.bottom_canvas)
        self.bottom_canvas.create_window((0, 0), window=self.bottom_inner, anchor="nw")
        self.bottom_inner.bind(
            "<Configure>",
            lambda e: self.bottom_canvas.configure(scrollregion=self.bottom_canvas.bbox("all"))
        )

        # Table
        self.table_frame = tk.Frame(self.bottom_inner, bg="#FEF2F4")
        self.table_frame.pack(anchor="center", pady=5)
        headers = ["Clicked RGB","Euclidean","KMeans","KNN","Random Forest", "Final RHS"]
        for col,header in enumerate(headers):
            tk.Label(self.table_frame,text=header,
                    font=("Arial",11,"bold"),
                    bg="#FEF2F4",relief="solid",borderwidth=1,width=20
            ).grid(row=0,column=col*2,padx=1,pady=1)

        # Column for color swatch
        tk.Label(self.table_frame,text="Color",
            font=("Arial",11,"bold"),
            bg="#FEF2F4",relief="solid",borderwidth=1,width=10
            ).grid(row=0,column=col*2+1,padx=1,pady=1)

    # -------------------- Load RHS colors --------------------
    def load_rhs(self):
        try:
            df = pd.read_excel("./data/color_codes_excel.xlsx", usecols=[0,1,2])
        except Exception as e:
            print("ERROR: Could not load Excel file.", e)
            return
        df.columns = df.columns.str.strip().str.lower()
        # Expect columns like "code","rgb","cluster"
        df = df.dropna(subset=["code","rgb","cluster"])
        df["rgb"] = df["rgb"].astype(str).str.strip()
        df = df[df["rgb"].str.contains(r"^\d+,\d+,\d+$")]
        df[["R","G","B"]] = df["rgb"].str.split(",",expand=True).astype(int)
        df = df[["code","rgb","R","G","B","cluster"]]
        self.rhs_colors = df.to_dict(orient="records")
        # also store arrays for quick use
        self.rhs_rgbs = np.array([[c["R"], c["G"], c["B"]] for c in self.rhs_colors])
        self.rhs_codes = [c["code"] for c in self.rhs_colors]
        print("RHS colors loaded:", len(self.rhs_colors))

    # -------------------- Train models --------------------
    def train_models(self):
        if not self.rhs_colors:
            print("No RHS colors loaded. ")
            return
        X = np.array([[c["R"],c["G"],c["B"]] for c in self.rhs_colors])
        self.X = X
        self.kmeans_model = KMeansCustom(n_clusters=10,max_iter=200,random_state=42).fit(X)
        self.knn_model = KNNSimple(k=3).fit(X,self.rhs_codes)

        # Train Random Forest for RHS code prediction (map codes -> ints)
        self.code_to_int = {code: i for i, code in enumerate(self.rhs_codes)}
        self.int_to_code = {v: k for k, v in self.code_to_int.items()}  # reversed properly
        y_int = np.array([self.code_to_int[c] for c in self.rhs_codes])
        self.rf_model = RandomForestClassifierSimple(n_estimators=20, max_depth=5, random_state=42).fit(X, y_int)

    # -------------------- Image upload --------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        # Load original image
        img = Image.open(file_path).convert("RGB")
        original_w, original_h = img.size

        # Get canvas size
        self.top_left_canvas.update_idletasks()
        cw = self.top_left_canvas.winfo_width() or 800
        ch = self.top_left_canvas.winfo_height() or 600

        # Compute scaling while keeping aspect ratio
        scale = min(max(cw / original_w, 0.01), max(ch / original_h, 0.01))

        # Final resized dimensions
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # Resize with preserved ratio
        self.img = img.resize((new_w, new_h), Image.LANCZOS)
        self.img_np = np.array(self.img)

        # Show original image in canvas
        self.image_preview = ImageTk.PhotoImage(self.img)
        self.show_centered(self.top_left_canvas, self.image_preview)

        # Corrected version using the resized np array
        #self.corrected_np = self.color_corrector(self.img_np)
        self.corrected_np = self.img_np
        corrected_img = Image.fromarray(self.corrected_np)

        # Display corrected image
        self.corrected_preview = ImageTk.PhotoImage(corrected_img)
        self.show_centered(self.top_right_canvas, self.corrected_preview)

    # -------------------- Center image --------------------
    def show_centered(self,canvas,image):
        canvas.update_idletasks()
        cw,ch = canvas.winfo_width(),canvas.winfo_height()
        iw,ih = image.width(),image.height()
        offset_x = (cw-iw)//2; offset_y = (ch-ih)//2
        canvas.delete("all")
        canvas.create_image(offset_x,offset_y,anchor="nw",image=image)

    # -------------------- Color correction --------------------
    def color_corrector(self,img_np):
        img = img_np.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l,a,b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        avg = np.mean(img,axis=(0,1))
        gray = np.mean(avg)
        scale = 1 + (gray-avg)/255*0.4
        img_bal = img.astype(np.float32)
        img_bal[...,0]*=scale[0]; img_bal[...,1]*=scale[1]; img_bal[...,2]*=scale[2]
        img = np.clip(img_bal,0,255).astype(np.uint8)
        gamma=0.95
        table=np.array([((i/255.0)**(1/gamma))*255 for i in range(256)]).astype(np.uint8)
        img = cv2.LUT(img,table)
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32)*1.08,0,255).astype(np.uint8)
        hsv=cv2.merge([h,s,v])
        img=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return img

    # -------------------- Find nearest RHS --------------------
    def find_nearest_rhs(self, rgb):
        rgb = np.array(rgb)
        dists = np.sqrt(np.sum((self.rhs_rgbs - rgb) ** 2, axis=1))
        idx = np.argmin(dists)
        return self.rhs_codes[idx]

    # -------------------- Pixel click --------------------
    def get_pixel_info(self,event):
        if not hasattr(self,"corrected_np"): return
        canvas = self.top_right_canvas
        canvas.update_idletasks()
        cw,ch = canvas.winfo_width(),canvas.winfo_height()
        ih,iw,_ = self.corrected_np.shape
        offset_x = (cw-iw)//2; offset_y = (ch-ih)//2
        if not (offset_x<=event.x<=offset_x+iw and offset_y<=event.y<=offset_y+ih): return
        px = event.x-offset_x; py = event.y-offset_y
        pixel = self.corrected_np[py,px]
        R,G,B = int(pixel[0]), int(pixel[1]), int(pixel[2])
        clicked_rgb = np.array([R,G,B])

        # Ask for true RHS
        popup = tk.Toplevel(self); popup.title("Input True RHS Code")
        tk.Label(popup,text=f"Clicked RGB: ({R},{G},{B})\nEnter true RHS code:").pack(padx=10,pady=10)
        code_entry = tk.Entry(popup); code_entry.pack(padx=10,pady=5)
        result = {}
        def submit(): result.update({"true_code":code_entry.get().strip()}); popup.destroy()
        tk.Button(popup,text="Submit",command=submit).pack(pady=5)
        self.wait_window(popup)
        if "true_code" not in result: return

        # Predictions
        rhs_euclid = min(self.rhs_colors,key=lambda c:(c["R"]-R)**2+(c["G"]-G)**2+(c["B"]-B)**2)

        # KMeans
        cluster = self.kmeans_model.predict(np.array([clicked_rgb]))[0]
        cluster_points = [c for i,c in enumerate(self.rhs_colors) if self.kmeans_model.labels_[i]==cluster]
        if cluster_points:
            rhs_kmeans = min(cluster_points,key=lambda c:(c["R"]-R)**2+(c["G"]-G)**2+(c["B"]-B)**2)
        else:
            rhs_kmeans = rhs_euclid  # fallback

        # KNN
        pred_knn_code = self.knn_model.predict(clicked_rgb)
        rhs_knn = next((c for c in self.rhs_colors if c["code"]==pred_knn_code), rhs_euclid)

        # Random Forest
        pred_int_rf = self.rf_model.predict(np.array([[R, G, B]]))[0]
        pred_code_rf = self.int_to_code.get(int(pred_int_rf), None)
        if pred_code_rf is None:
            # fallback to nearest in RGB space
            pred_code_rf = self.find_nearest_rhs([R,G,B])
        rhs_rf = next((c for c in self.rhs_colors if c["code"] == pred_code_rf), rhs_euclid)

        self.clicked_pixels.append({
            "clicked_rgb":clicked_rgb,
            "true_rhs_code":result["true_code"],
            "predictions":{"Euclidean":rhs_euclid,"KMeans":rhs_kmeans,"KNN":rhs_knn,"Random Forest":rhs_rf}
        })
        self.show_pixel_result(clicked_rgb,rhs_euclid,rhs_kmeans,rhs_knn,rhs_rf)
        
    # -------------------- Shows pixel result --------------------
    def show_pixel_result(self, rgb, rhs_euclid, rhs_kmeans, rhs_knn, rhs_rf):
        R, G, B = rgb
        row = len(self.clicked_pixels)

        preds = [rhs_euclid, rhs_kmeans, rhs_knn, rhs_rf, rhs_kmeans]

        # Clicked RGB cell and swatch
        tk.Label(self.table_frame, text=f"{R},{G},{B}",
             bg="white", relief="solid", borderwidth=1, width=20
        ).grid(row=row, column=0, padx=1, pady=1)

        swatch = tk.Frame(self.table_frame, bg=f"#{R:02x}{G:02x}{B:02x}", width=40, height=20)
        swatch.grid(row=row, column=1, padx=1, pady=1)
        swatch.grid_propagate(False)

        col = 2
        for i, pred in enumerate(preds):
            code = pred.get("code","?")
            pr, pg, pb = int(pred.get("R",0)), int(pred.get("G",0)), int(pred.get("B",0))
            
            font_to_use = ("Arial", 12, "bold") if i == len(preds)-1 else ("Arial", 11)
            
            tk.Label(self.table_frame,
                text=f"{code} ({pr},{pg},{pb})",
                bg="white", relief="solid", borderwidth=1, width=20, font=font_to_use
            ).grid(row=row, column=col, padx=1, pady=1)

            swatch = tk.Frame(self.table_frame,
                          bg=f"#{pr:02x}{pg:02x}{pb:02x}",
                          width=40, height=20)
            swatch.grid(row=row, column=col+1, padx=1, pady=1)
            swatch.grid_propagate(False)

            col += 2

    # -------------------- Accuracy report --------------------
    def generate_report(self):
        if not self.clicked_pixels:
            messagebox.showinfo("Info","No pixels clicked yet!"); return
        max_dist = np.sqrt(255**2+255**2+255**2)
        report={}
        for method in ["Euclidean","KMeans","KNN","Random Forest"]:
            dist_list,mae_list,mse_list,sim_list=[],[],[],[]
            for item in self.clicked_pixels:
                pred = item["predictions"][method]
                pred_rgb = np.array([pred["R"],pred["G"],pred["B"]])
                true_code = item["true_rhs_code"]
                true_candidates = [c for c in self.rhs_colors if c["code"]==true_code]
                if not true_candidates:
                    # skip this item if the entered true_code can't be found
                    continue
                true = true_candidates[0]
                true_rgb = np.array([true["R"],true["G"],true["B"]])
                dist = np.linalg.norm(pred_rgb-true_rgb)
                dist_list.append(dist)
                mae_list.append(np.mean(np.abs(pred_rgb-true_rgb)))
                mse_list.append(np.mean((pred_rgb-true_rgb)**2))
                sim_list.append(100*(1-dist/max_dist))
            if not dist_list:
                report[method] = {
                    "Average Euclidean Distance": float("nan"),
                    "Average RGB Similarity %": float("nan"),
                    "MAE per channel": float("nan"),
                    "MSE per channel": float("nan")
                }
            else:
                report[method] = {
                    "Average Euclidean Distance": np.mean(dist_list),
                    "Average RGB Similarity %": np.mean(sim_list),
                    "MAE per channel": np.mean(mae_list),
                    "MSE per channel": np.mean(mse_list)
                }
        
        # Show popup
        popup = tk.Toplevel(self); popup.title("Accuracy Report"); popup.geometry("450x350")
        tk.Label(popup,text="Accuracy Report (RGB Metrics)",font=("Arial",14,"bold")).pack(pady=10)
        for method,metrics in report.items():
            tk.Label(popup,text=f"{method}:",font=("Arial",12,"bold")).pack(anchor="w",padx=10)
            tk.Label(popup,text=f"  Avg Euclidean Distance: {metrics['Average Euclidean Distance']:.2f}",font=("Arial",11)).pack(anchor="w",padx=20)
            tk.Label(popup,text=f"  Avg RGB Similarity: {metrics['Average RGB Similarity %']:.2f}%",font=("Arial",11)).pack(anchor="w",padx=20)
            tk.Label(popup,text=f"  MAE per channel: {metrics['MAE per channel']:.2f}",font=("Arial",11)).pack(anchor="w",padx=20)
            tk.Label(popup,text=f"  MSE per channel: {metrics['MSE per channel']:.2f}",font=("Arial",11)).pack(anchor="w",padx=20)
    
    # -------------------- Returns to previous screen --------------------
    def go_back(self):
        self.destroy()
        self.master.deiconify()

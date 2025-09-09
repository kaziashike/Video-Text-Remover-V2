import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
import os
import threading
import time
import webbrowser

class VideoSubtitleRemoverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Subtitle Remover")
        self.root.geometry("600x700")
        
        # API Configuration
        self.api_url = "http://localhost:8000"
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.use_h264 = tk.BooleanVar(value=True)
        self.mode = tk.StringVar(value="sttn")
        self.threshold_height_width_difference = tk.IntVar(value=10)
        self.subtitle_area_deviation_pixel = tk.IntVar(value=20)
        self.threshold_height_difference = tk.IntVar(value=20)
        self.pixel_tolerance_y = tk.IntVar(value=20)
        self.pixel_tolerance_x = tk.IntVar(value=20)
        self.sttn_skip_detection = tk.BooleanVar(value=True)
        self.sttn_neighbor_stride = tk.IntVar(value=5)
        self.sttn_reference_length = tk.IntVar(value=10)
        self.sttn_max_load_num = tk.IntVar(value=50)
        self.propainter_max_load_num = tk.IntVar(value=70)
        self.lama_super_fast = tk.BooleanVar(value=False)
        
        self.task_id = None
        self.check_status_job = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # API Configuration
        api_frame = ttk.LabelFrame(main_frame, text="API Configuration", padding="10")
        api_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(api_frame, text="API URL:").grid(row=0, column=0, sticky=tk.W)
        self.api_url_entry = ttk.Entry(api_frame, width=40)
        self.api_url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        self.api_url_entry.insert(0, self.api_url)
        ttk.Button(api_frame, text="Check API", command=self.check_api).grid(row=0, column=2, padx=(5, 0))
        
        # File Selection
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_video).grid(row=0, column=2)
        
        # Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Mode selection
        ttk.Label(config_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        mode_combo = ttk.Combobox(config_frame, textvariable=self.mode, values=["sttn", "lama", "propainter"], state="readonly")
        mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Checkboxes
        ttk.Checkbutton(config_frame, text="Use H264", variable=self.use_h264).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(config_frame, text="STTN Skip Detection", variable=self.sttn_skip_detection).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(config_frame, text="LAMA Super Fast", variable=self.lama_super_fast).grid(row=1, column=2, sticky=tk.W)
        
        # Parameters
        params = [
            ("Threshold Height Width Difference:", self.threshold_height_width_difference, 0),
            ("Subtitle Area Deviation Pixel:", self.subtitle_area_deviation_pixel, 1),
            ("Threshold Height Difference:", self.threshold_height_difference, 2),
            ("Pixel Tolerance Y:", self.pixel_tolerance_y, 3),
            ("Pixel Tolerance X:", self.pixel_tolerance_x, 4),
            ("STTN Neighbor Stride:", self.sttn_neighbor_stride, 5),
            ("STTN Reference Length:", self.sttn_reference_length, 6),
            ("STTN Max Load Num:", self.sttn_max_load_num, 7),
            ("PROPAINTER Max Load Num:", self.propainter_max_load_num, 8),
        ]
        
        for i, (label, variable, row) in enumerate(params):
            ttk.Label(config_frame, text=label).grid(row=row+2, column=0, sticky=tk.W)
            ttk.Entry(config_frame, textvariable=variable, width=10).grid(row=row+2, column=1, sticky=tk.W, padx=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        self.process_button = ttk.Button(button_frame, text="Process Video", command=self.process_video)
        self.process_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_button = ttk.Button(button_frame, text="Check Status", command=self.check_status, state=tk.DISABLED)
        self.status_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.download_button = ttk.Button(button_frame, text="Download Result", command=self.download_result, state=tk.DISABLED)
        self.download_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(1, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        
    def check_api(self):
        try:
            response = requests.get(f"{self.api_url_entry.get()}/health", timeout=5)
            if response.status_code == 200:
                messagebox.showinfo("API Check", "API is running and accessible!")
                self.api_url = self.api_url_entry.get()
            else:
                messagebox.showerror("API Check", f"API returned status code: {response.status_code}")
        except Exception as e:
            messagebox.showerror("API Check", f"Failed to connect to API: {str(e)}")
    
    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path.set(file_path)
            
    def process_video(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
            
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Selected video file does not exist")
            return
            
        # Start processing in a separate thread
        self.process_button.config(state=tk.DISABLED)
        self.progress_var.set("Uploading video...")
        self.progress_bar['value'] = 0
        
        thread = threading.Thread(target=self._process_video_thread)
        thread.daemon = True
        thread.start()
        
    def _process_video_thread(self):
        try:
            # Prepare files and data
            with open(self.video_path.get(), 'rb') as video_file:
                files = {'video': (os.path.basename(self.video_path.get()), video_file, 'video/mp4')}
                data = {
                    'mode': self.mode.get(),
                    'use_h264': self.use_h264.get(),
                    'threshold_height_width_difference': self.threshold_height_width_difference.get(),
                    'subtitle_area_deviation_pixel': self.subtitle_area_deviation_pixel.get(),
                    'threshold_height_difference': self.threshold_height_difference.get(),
                    'pixel_tolerance_y': self.pixel_tolerance_y.get(),
                    'pixel_tolerance_x': self.pixel_tolerance_x.get(),
                    'sttn_skip_detection': self.sttn_skip_detection.get(),
                    'sttn_neighbor_stride': self.sttn_neighbor_stride.get(),
                    'sttn_reference_length': self.sttn_reference_length.get(),
                    'sttn_max_load_num': self.sttn_max_load_num.get(),
                    'propainter_max_load_num': self.propainter_max_load_num.get(),
                    'lama_super_fast': self.lama_super_fast.get()
                }
                
                response = requests.post(f"{self.api_url}/process", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                self.task_id = result['task_id']
                self.progress_var.set("Processing started")
                self.status_button.config(state=tk.NORMAL)
                
                # Start checking status
                self._check_status()
            else:
                self.progress_var.set(f"Error: {response.status_code}")
                self.process_button.config(state=tk.NORMAL)
                
        except Exception as e:
            self.progress_var.set(f"Error: {str(e)}")
            self.process_button.config(state=tk.NORMAL)
            
    def check_status(self):
        if not self.task_id:
            return
            
        self.status_button.config(state=tk.DISABLED)
        self._check_status()
        
    def _check_status(self):
        try:
            response = requests.get(f"{self.api_url}/status/{self.task_id}")
            if response.status_code == 200:
                status_data = response.json()
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                self.progress_var.set(f"Status: {status}")
                self.progress_bar['value'] = progress
                
                if status == "processing":
                    # Schedule next check
                    self.check_status_job = self.root.after(2000, self._check_status)
                elif status == "completed":
                    self.progress_var.set("Processing completed!")
                    self.download_button.config(state=tk.NORMAL)
                    self.process_button.config(state=tk.NORMAL)
                elif status == "failed":
                    error = status_data.get('error', 'Unknown error')
                    self.progress_var.set(f"Processing failed: {error}")
                    self.process_button.config(state=tk.NORMAL)
            else:
                self.progress_var.set(f"Error checking status: {response.status_code}")
                self.process_button.config(state=tk.NORMAL)
                
        except Exception as e:
            self.progress_var.set(f"Error checking status: {str(e)}")
            self.process_button.config(state=tk.NORMAL)
            
    def download_result(self):
        if not self.task_id:
            return
            
        try:
            # Get the output filename
            response = requests.get(f"{self.api_url}/status/{self.task_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data['status'] == "completed":
                    # Ask user where to save the file
                    output_file = filedialog.asksaveasfilename(
                        title="Save processed video",
                        defaultextension=".mp4",
                        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
                    )
                    
                    if output_file:
                        # Download the file
                        download_response = requests.get(f"{self.api_url}/download/{self.task_id}")
                        if download_response.status_code == 200:
                            with open(output_file, 'wb') as f:
                                f.write(download_response.content)
                            messagebox.showinfo("Download", f"File saved to {output_file}")
                        else:
                            messagebox.showerror("Download", f"Failed to download file: {download_response.status_code}")
                else:
                    messagebox.showerror("Download", "Task is not completed yet")
            else:
                messagebox.showerror("Download", f"Failed to get task status: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Download", f"Error downloading file: {str(e)}")

def main():
    root = tk.Tk()
    app = VideoSubtitleRemoverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
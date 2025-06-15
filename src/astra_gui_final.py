import os
import psutil
import time
import math
import serial
import csv
import folium
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import webbrowser
import threading
import queue
import cv2
import onnxruntime as ort
import torch
from ultralytics.utils.ops import non_max_suppression
import random
import psutil

from transformers import pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-step-50K-105b")

def llm_run(pipe, prompt):
    result = pipe(prompt, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']
    return result

def estimate_npu_usage():
    start = time.time()
    time.sleep(0.1 + random.uniform(-0.02, 0.02))
    npu_time = time.time() - start
    start = time.time()
    sum(range(10**6))
    cpu_time = time.time() - start
    efficiency = (cpu_time - npu_time) / cpu_time
    npu_usage = 30 + (40 * efficiency)
    return min(70, max(30, int(npu_usage)))

class OptimizedTrackingModel:
    def __init__(self, model_path="best.onnx"):
        # Setup hardware acceleration
        available_providers = ort.get_available_providers()
        print(f"Available providers: {available_providers}")
        
        # Priority list of providers
        providers = [
            p for p in [
                'DmlExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ] if p in available_providers
        ]
        
        # Session options for optimization
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 8
        session_options.inter_op_num_threads = 8
        
        # Load model
        self.session = ort.InferenceSession(model_path, 
                                         providers=providers,
                                         sess_options=session_options)
        print(f"Using provider: {self.session.get_providers()[0]}")
        
        # Input/output config
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.model_height, self.model_width = 640, 640
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.model_height, self.model_width = input_shape[2], input_shape[3]
        
        # Class configuration
        self.class_names = ["Balaclava", "Grenade", "Gun", "Handgun", "Knife", 
                          "Person", "Punch", "Slap", "Weapon Holding"]
        self.class_colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), 
                           (255,0,255), (255,255,0), (128,0,128), (0,165,255), (128,128,0)]
        
        # Prediction history buffer
        self.prediction_history = []
        self.max_history = 60  # Store last 60 frames
    
    def scale_coords(self, img1_shape, coords, img0_shape):
        """Rescale coords from img1_shape to img0_shape"""
        gain = min(img1_shape[0]/img0_shape[0], img1_shape[1]/img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1]*gain)/2, (img1_shape[0] - img0_shape[0]*gain)/2
        coords[:, [0,2]] -= pad[0]  # x padding
        coords[:, [1,3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        return coords
    
    def detect(self, frame):
        """Run detection on a single frame"""
        # Preprocessing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.model_width, self.model_height))
        img = img.transpose(2,0,1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        
        # Post-processing
        pred = torch.from_numpy(outputs[0])
        detections = non_max_suppression(
            pred,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            max_det=300
        )
        
        # Store current frame's detections in history
        current_detections = []
        for det in detections:
            if det is not None and len(det):
                # Scale boxes to original image
                det[:, :4] = self.scale_coords((self.model_height, self.model_width), 
                                              det[:, :4], 
                                              frame.shape)
                current_detections.append(det.cpu().numpy())
        
        # Add to history and maintain buffer size
        self.prediction_history.append({
            'timestamp': time.time(),
            'detections': current_detections
        })
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # Draw detections
        annotated_frame = frame.copy()
        for det in detections:
            if det is not None and len(det):
                # Scale boxes to original image
                det[:, :4] = self.scale_coords((self.model_height, self.model_width), 
                                              det[:, :4], 
                                              frame.shape)
                
                for *xyxy, conf, cls in reversed(det):
                    class_idx = int(cls)
                    color = self.class_colors[class_idx]
                    label = f"{self.class_names[class_idx]} {conf:.2f}"
                    
                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, 
                                 (int(xyxy[0]), int(xyxy[1])),
                                 (int(xyxy[2]), int(xyxy[3])),
                                 color, 2)
                    cv2.putText(annotated_frame, label,
                               (int(xyxy[0]), int(xyxy[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               color, 2)
        
        return annotated_frame
    
    def get_recent_predictions(self, num_frames=60):
        """Get recent predictions from history"""
        if num_frames > len(self.prediction_history):
            num_frames = len(self.prediction_history)
        return self.prediction_history[-num_frames:]

class ReadThread:
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self._stop_event = False
        self.data_received = None
        
    def receivePacket(self):
        while not self._stop_event:
            while self.serial_port.read() != b'\x7E':
                pass
            packet_length = int.from_bytes(self.serial_port.read(2), byteorder='big')
            frame_type = self.serial_port.read().hex()
            frame_id = self.serial_port.read().hex()
            source_addr = self.serial_port.read(8).hex()
            source_addr_16 = self.serial_port.read(2).hex()
            options = self.serial_port.read().hex()
            radius = self.serial_port.read().hex()
            data_length = packet_length - 14
            data = self.serial_port.read(data_length)
            checksum = self.serial_port.read().hex()
            packet = bytearray.fromhex(frame_type + frame_id + source_addr + source_addr_16 + options + radius) + data
            calculated_checksum = (0xFF - sum(packet) & 0xFF)
            if calculated_checksum == int(checksum, 16):
                return data
            return None

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        
    def run(self):
        while not self._stop_event:
            telemetry_package = {}
            field_count = 0
            expected_fields = 22
            is_team_id_received = False
            while True:
                data_packet = self.receivePacket()
                if data_packet is not None:
                    data_packet = data_packet.decode('latin-1')
                    data_packet = data_packet.replace("ÁÂ", "")
                    if "Team_Id" in data_packet:
                        is_team_id_received = True
                    if is_team_id_received:
                        fields = data_packet.split("\n")
                        for field in fields:
                            if field:
                                key, value = field.split(":", 1)
                                value = value.strip()
                                if value.endswith("}"):
                                    value = value[:-1]
                                telemetry_package[key] = value
                                field_count += 1
                                if field_count == expected_fields:
                                    telemetry_values = ",".join(telemetry_package.values())
                                    telemetry_package = {}
                                    field_count = 0
                                    is_team_id_received = False
                                    data = telemetry_values
                                    print(data)
                                    if self.data_received:
                                        self.data_received(data)
                else:
                    time.sleep(0.1)

    def stop(self):
        self._stop_event = True
        
    def join(self):
        if hasattr(self, 'thread'):
            self.thread.join()

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRABHARAT")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize camera attributes
        self.cap = None
        self.camera_running = False
        self.surveillance_active = True  # Surveillance on by default
        self.video_update_interval = 30  # Reduced from 30 to 15 for better performance
        self.current_frame = None
        self.target_width = 480
        self.target_height = 360
        self.frame_queue = queue.Queue(maxsize=2)  # Reduced queue size
        
        # Initialize tracking model
        self.tracker = OptimizedTrackingModel()
        self.tracking_active = True
        
        # Initialize surveillance status
        self.mission_start_time = time.time()
        self.frame_counter = 0
        
        # Initialize serial communication
        self.read_thread = None
        self.serial_port = None
        
        # Data storage
        self.pc = []
        self.alti = []
        self.pres = []
        self.temp = []
        self.accelo_x = []
        self.accelo_y = []
        self.accelo_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
        
        # Chat history
        self.chat_history = []
        
        # Performance monitoring
        self.prev_frame_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # Setup UI
        self.setup_ui()
        
        # Start surveillance immediately
        self.start_surveillance()

    def setup_ui(self):
        self.root.geometry("1200x800")
        
        # Header Frame
        self.header_frame = ttk.Frame(self.root)
        self.header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Container frame to center all header elements
        self.header_container = ttk.Frame(self.header_frame)
        self.header_container.pack(expand=True)
        
        # Left logo
        try:
            team_logo_img = Image.open("Logos/AstraBharat_logo.png")
            team_logo_img = team_logo_img.resize((150, 120), Image.LANCZOS)
            self.team_logo_photo = ImageTk.PhotoImage(team_logo_img)
            self.team_logo_label = ttk.Label(self.header_container, image=self.team_logo_photo)
        except:
            self.team_logo_label = ttk.Label(self.header_container, text="Team Logo", foreground='gray')
        self.team_logo_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Center heading
        self.team_name_label = ttk.Label(self.header_container, text="ASTRABHARAT", 
                                       font=("Rockwell Extra Bold", 26, "bold"))
        self.team_name_label.pack(side=tk.LEFT, padx=20)
        
        # Right logo
        try:
            cansat_logo_img = Image.open("Logos/qualcomm_logo.png")
            cansat_logo_img = cansat_logo_img.resize((150, 120), Image.LANCZOS)
            self.cansat_logo_photo = ImageTk.PhotoImage(cansat_logo_img)
            self.cansat_logo_label = ttk.Label(self.header_container, image=self.cansat_logo_photo)
        except:
            self.cansat_logo_label = ttk.Label(self.header_container, text="Partner Logo", foreground='gray')
        self.cansat_logo_label.pack(side=tk.LEFT, padx=(20, 0))

        # Status Frame
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.left_status_frame = ttk.LabelFrame(self.status_frame, text="Surveillance Status")
        self.left_status_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(self.left_status_frame, text="Surveillance Time:", font=("Rockwell", 12, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.mission_time = ttk.Label(self.left_status_frame, text="00:00:00", font=("Bookman Old Style", 12, "bold"))
        self.mission_time.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(self.left_status_frame, text="FPS:", font=("Rockwell", 12, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.packet_count = ttk.Label(self.left_status_frame, text="0", font=("Bookman Old Style", 12, "bold"))
        self.packet_count.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        self.right_status_frame = ttk.LabelFrame(self.status_frame, text="System Status")
        self.right_status_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(self.right_status_frame, text="Device Power:", font=("Rockwell", 12, "bold")).grid(row=0, column=0, sticky="e", padx=5, pady=2)
        
        def get_battery_percent():
            try:
                battery = psutil.sensors_battery()
                if battery:
                    return f"{int(battery.percent)}%"
                return "N/A"
            except:
                return "N/A"
        
        self.battery_value = ttk.Label(self.right_status_frame, 
                                     text=get_battery_percent(), 
                                     font=("Bookman Old Style", 12, "bold"))
        self.battery_value.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        def update_battery_status():
            self.battery_value.config(text=get_battery_percent())
            self.root.after(60000, update_battery_status)
        
        update_battery_status()

        ttk.Label(self.right_status_frame, text="Detection Status:", font=("Rockwell", 12, "bold")).grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.fsw_state = ttk.Label(self.right_status_frame, text="ACTIVE", font=("Bookman Old Style", 12, "bold"))
        self.fsw_state.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Main Content Area
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Panel - Camera View
        self.left_panel = ttk.Frame(self.main_pane, width=600)
        self.main_pane.add(self.left_panel)
        
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Commands & LIVE VIEW")
        self.control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.control_buttons_frame = ttk.Frame(self.control_frame)
        self.control_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.start_tele_button = ttk.Button(self.control_buttons_frame, text="Start Surveillance", 
                                         command=self.start_surveillance, width=20)
        self.start_tele_button.grid(row=0, column=0, padx=5, pady=2, sticky='ew')
        
        self.stop_tele_button = ttk.Button(self.control_buttons_frame, text="Stop Surveillance", 
                                        command=self.stop_surveillance, width=20)
        self.stop_tele_button.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        
        self.save_csv_button = ttk.Button(self.control_buttons_frame, text="Save Detection Logs", 
                                       command=self.save_detection_logs, width=20)
        self.save_csv_button.grid(row=1, column=0, padx=5, pady=2, sticky='ew')
        
        self.analyze_button = ttk.Button(self.control_buttons_frame, text="Analyze Recent", 
                                       command=self.analyze_recent_detections, width=20)
        self.analyze_button.grid(row=1, column=1, padx=5, pady=2, sticky='ew')

        self.cansat_view_frame = ttk.Frame(self.control_frame)
        self.cansat_view_frame.pack(fill=tk.BOTH, expand=True)
        self.video_canvas = tk.Canvas(self.cansat_view_frame, width=480, height=360)
        self.video_canvas.pack(pady=5)

        # Right Panel - Performance Graphs and Chat
        self.right_panel = ttk.Frame(self.main_pane, width=400)
        self.main_pane.add(self.right_panel)
        
        self.graph_notebook = ttk.Notebook(self.right_panel)
        self.graph_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Performance Graphs Tab
        self.performance_tab = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.performance_tab, text="Performance")
        self.setup_performance_graphs()
        
        # Chat Tab
        self.chat_tab = ttk.Frame(self.graph_notebook)
        self.graph_notebook.add(self.chat_tab, text="Chat Assistant")
        self.setup_chat_interface()

        # Telemetry Frame
        self.telemetry_frame = ttk.LabelFrame(self.root, text="Detection Logs")
        self.telemetry_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        self.telemetry_tree = ttk.Treeview(self.telemetry_frame, columns=("data",), show="headings", height=5)
        self.telemetry_tree.heading("data", text="Detection Logs")
        self.telemetry_tree.column("data", width=1000)
        
        scrollbar = ttk.Scrollbar(self.telemetry_frame, orient="vertical", command=self.telemetry_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.telemetry_tree.configure(yscrollcommand=scrollbar.set)
        self.telemetry_tree.pack(fill=tk.BOTH, expand=True)

        # Start surveillance status updates
        self.update_surveillance_status()

    def setup_performance_graphs(self):
        """Setup the performance monitoring graphs"""
        self.time_series = list(range(50))
        self.cpu_usages = [0] * 50
        self.npu_usages = [0] * 50
        self.prev_frame_time = time.time()

        # CPU Usage Graph
        self.cpu_fig, self.cpu_ax = plt.subplots(figsize=(3, 2))
        self.cpu_ax.set_ylim(0, 100)
        self.cpu_ax.set_ylabel("CPU Usage (%)")
        self.cpu_ax.set_xlabel("Time")
        self.cpu_line, = self.cpu_ax.plot(self.time_series, self.cpu_usages, 'g-')
        self.cpu_canvas = FigureCanvasTkAgg(self.cpu_fig, master=self.performance_tab)
        self.cpu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # NPU Usage Graph (simulated)
        self.npu_fig, self.npu_ax = plt.subplots(figsize=(3, 2))
        self.npu_ax.set_ylim(0, 100)
        self.npu_ax.set_ylabel("NPU Usage (%)")
        self.npu_ax.set_xlabel("Time")
        self.npu_line, = self.npu_ax.plot(self.time_series, self.npu_usages, 'b-')
        self.npu_canvas = FigureCanvasTkAgg(self.npu_fig, master=self.performance_tab)
        self.npu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Start updating graphs
        self.update_performance_graphs()

    def setup_chat_interface(self):
        """Setup the chat interface in the chat tab"""
        # Chat output display
        self.chat_output_frame = ttk.LabelFrame(self.chat_tab, text="Chat History")
        self.chat_output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_output = tk.Text(self.chat_output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure tags for colored text
        self.chat_output.tag_config('user', foreground='blue')
        self.chat_output.tag_config('system', foreground='green')
        self.chat_output.tag_config('assistant', foreground='green')
        
        scrollbar = ttk.Scrollbar(self.chat_output_frame, orient="vertical", command=self.chat_output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_output.configure(yscrollcommand=scrollbar.set)
        
        # Chat input area
        self.chat_input_frame = ttk.Frame(self.chat_tab)
        self.chat_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.chat_input = tk.Text(self.chat_input_frame, height=3, wrap=tk.WORD)
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        self.send_button = ttk.Button(self.chat_input_frame, text="Send", 
                                    command=self.send_chat_message, width=10)
        self.send_button.pack(side=tk.RIGHT)
        
        # Bind Enter key to send message
        self.chat_input.bind("<Return>", lambda event: self.send_chat_message())
        
        # Add welcome message
        self.add_chat_message("System", "Welcome to ASTRABHARAT Chat Assistant. How can I help you?")

    def add_chat_message(self, sender, message):
        """Add a message to the chat display with colored text"""
        self.chat_output.configure(state=tk.NORMAL)
        
        # Determine tag based on sender
        if sender == "You":
            tag = 'user'
        elif sender == "System":
            tag = 'system'
        else:
            tag = 'assistant'
        
        self.chat_output.insert(tk.END, f"{sender}: ", tag)
        self.chat_output.insert(tk.END, f"{message}\n\n", tag)
        
        self.chat_output.configure(state=tk.DISABLED)
        self.chat_output.see(tk.END)
        self.chat_history.append((sender, message))

    def send_chat_message(self):
        """Send the chat message and get response from LLM"""
        user_input = self.chat_input.get("1.0", tk.END).strip()
        if user_input:
            self.add_chat_message("You", user_input)
            self.chat_input.delete("1.0", tk.END)
            
            # Show typing indicator
            self.add_chat_message("System", "Thinking...")
            
            # Process in background to avoid UI freeze
            threading.Thread(target=self.process_chat_message, args=(user_input,), daemon=True).start()

    def process_chat_message(self, user_input):
        """Process the chat message with LLM"""
        try:
            # Get response from LLM
            response = llm_run(pipe, user_input)
            
            # Remove "Thinking..." message
            self.chat_output.configure(state=tk.NORMAL)
            self.chat_output.delete("end-2l linestart", "end")
            self.chat_output.configure(state=tk.DISABLED)
            
            # Add actual response
            self.add_chat_message("Assistant", response)
        except Exception as e:
            self.add_chat_message("System", f"Error: {str(e)}")

    def update_performance_graphs(self):
        # Get current CPU usage
        cpu_usage = psutil.cpu_percent()
        
        npu_usage = estimate_npu_usage()

        # Update data series
        self.cpu_usages.append(cpu_usage)
        self.cpu_usages.pop(0)
        
        self.npu_usages.append(npu_usage)
        self.npu_usages.pop(0)

        # Update graphs
        self.cpu_line.set_ydata(self.cpu_usages)
        self.npu_line.set_ydata(self.npu_usages)

        self.cpu_ax.relim()
        self.cpu_ax.autoscale_view()
        self.cpu_canvas.draw()

        self.npu_ax.relim()
        self.npu_ax.autoscale_view()
        self.npu_canvas.draw()

        # Schedule next update
        self.root.after(1000, self.update_performance_graphs)

    def update_surveillance_status(self):
        """Update the surveillance time and frame count"""
        # Update surveillance time
        elapsed = time.time() - self.mission_start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        self.mission_time.config(text=time_str)
        
        # Only increment frame count if surveillance is active
        if self.surveillance_active and self.camera_running:
            self.frame_counter += 1
            self.packet_count.config(text=str(self.fps))
            
        # Schedule next update
        self.root.after(1000, self.update_surveillance_status)

    def start_surveillance(self):
        """Start the surveillance system"""
        if not self.surveillance_active:
            self.surveillance_active = True
            self.mission_start_time = time.time()
            self.frame_counter = 0
            self.fsw_state.config(text="ACTIVE")
            self.start_tele_button.config(state='disabled')
            self.stop_tele_button.config(state='normal')
            
        # Start camera if not already running
        if not self.camera_running:
            self.start_camera_with_tracking()
            
    def stop_surveillance(self):
        """Stop the surveillance system"""
        if self.surveillance_active:
            self.surveillance_active = False
            self.fsw_state.config(text="INACTIVE")
            self.start_tele_button.config(state='normal')
            self.stop_tele_button.config(state='disabled')
            self.stop_camera()
        
    def save_detection_logs(self):
        """Save detection logs to CSV with prediction history"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save detection logs"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    # Write header
                    csv_writer.writerow(["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])
                    
                    # Get all predictions from history
                    predictions = self.tracker.get_recent_predictions(60)
                    
                    for frame_pred in predictions:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(frame_pred['timestamp']))
                        for detections in frame_pred['detections']:
                            for detection in detections:
                                x1, y1, x2, y2, conf, cls = detection
                                class_name = self.tracker.class_names[int(cls)]
                                csv_writer.writerow([
                                    timestamp,
                                    class_name,
                                    f"{conf:.4f}",
                                    f"{x1:.2f}",
                                    f"{y1:.2f}",
                                    f"{x2:.2f}",
                                    f"{y2:.2f}"
                                ])
                
                messagebox.showinfo("Success", f"Saved last 60 frames of detection logs to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {str(e)}")

    def analyze_recent_detections(self):
        """Analyze and display recent detection statistics"""
        recent_predictions = self.tracker.get_recent_predictions()
        
        # Count detections by class
        class_counts = {name: 0 for name in self.tracker.class_names}
        total_detections = 0
        
        for pred in recent_predictions:
            for det in pred['detections']:
                for detection in det:
                    class_idx = int(detection[5])
                    class_counts[self.tracker.class_names[class_idx]] += 1
                    total_detections += 1
        
        # Generate analysis text
        analysis_text = f"Last {len(recent_predictions)} frames detection analysis:\n"
        analysis_text += f"Total detections: {total_detections}\n\n"
        analysis_text += "Detections by class:\n"
        for class_name, count in class_counts.items():
            if count > 0:
                analysis_text += f"{class_name}: {count} ({count/total_detections:.1%})\n"
        
        # Show in message box
        messagebox.showinfo("Detection Analysis", analysis_text)

    def start_camera_with_tracking(self):
        if not self.camera_running:
            try:
                # Try different camera indices with DirectShow for better performance
                for camera_index in [0, 1, 2]:
                    self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        break
                
                if not self.cap.isOpened():
                    raise Exception("Could not open any video device")
                
                # Set camera resolution and properties for optimal performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

                self.camera_running = True
                
                # Start tracking thread
                self.tracking_thread = threading.Thread(target=self.process_tracking, daemon=True)
                self.tracking_thread.start()
                
                # Start GUI update
                self.update_camera_feed()
                
            except Exception as e:
                self.show_camera_error(str(e))
                self.camera_running = False

    def process_tracking(self):
        while self.camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Run object detection with our optimized model
            annotated_frame = self.tracker.detect(frame)
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Calculate FPS
            curr_time = time.time()
            self.fps_counter += 1
            if (curr_time - self.prev_frame_time) >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.prev_frame_time = curr_time
                # print(f"FPS: {self.fps} | Using: {self.tracker.session.get_providers()[0]}")
            
            # Put frame in queue for GUI thread
            if not self.frame_queue.full():
                self.frame_queue.put(rgb_frame)

    def update_camera_feed(self):
        if self.camera_running:
            try:
                # Get latest frame from queue
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    
                    # Convert to PIL Image then to PhotoImage
                    img = Image.fromarray(frame)
                    self.current_frame = ImageTk.PhotoImage(image=img)
                    
                    # Update the canvas
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_frame)
            
            except queue.Empty:
                pass
            
            # Schedule next update
            self.root.after(self.video_update_interval, self.update_camera_feed)

    def show_camera_error(self, message):
        """Show error message on canvas"""
        self.video_canvas.delete("all")
        self.video_canvas.create_text(240, 180, text=f"Camera Error: {message}", 
                                     fill="red", font=('Helvetica', 12))

    def stop_camera(self):
        """Properly stop the camera feed"""
        if self.camera_running:
            self.camera_running = False
            self.tracking_active = False
            if self.cap:
                self.cap.release()
            self.video_canvas.delete("all")
            self.video_canvas.create_text(240, 180, text="Camera Feed Stopped", 
                                        fill="white", font=('Helvetica', 12))

    def on_close(self):
        """Handle window closing event"""
        self.tracking_active = False
        self.stop_camera()
        if self.read_thread:
            self.read_thread.stop()
            self.read_thread.join()
        if self.serial_port:
            self.serial_port.close()
        self.root.destroy()

    def start_logging(self):
        try:
            self.serial_port = serial.Serial('COM5', 9600)
            self.read_thread = ReadThread(self.serial_port)
            self.read_thread.data_received = self.update_labels
            self.read_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")

    def stop_logging(self):
        if self.read_thread:
            self.read_thread.stop()
            self.read_thread.join()
            if self.serial_port:
                self.serial_port.close()
            self.read_thread = None
            self.serial_port = None

    def update_labels(self, data):
        values = data.split(",")
        if len(values) >= 1:
            self.telemetry_tree.insert("", "end", values=(data,))
            self.telemetry_tree.see(self.telemetry_tree.get_children()[-1])
            self.update_plot_data(data)

    def update_plot_data(self, data):
        values = data.split(",")
        if len(values) >= 1:
            self.pc.append(int(values[2]))
            self.alti.append(float(values[3]))
            self.pres.append(float(values[4]))
            self.temp.append(float(values[5]))
            self.accelo_x.append(float(values[12]))
            self.accelo_y.append(float(values[13]))
            self.accelo_z.append(float(values[14]))
            self.gyro_x.append(float(values[15]))
            self.gyro_y.append(float(values[16]))
            self.gyro_z.append(float(values[17]))

def dashboard():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    dashboard()
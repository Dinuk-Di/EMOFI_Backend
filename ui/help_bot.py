import re
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import sys   
import os     

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to ui/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # path to DesktopApp/
sys.path.append(PROJECT_ROOT)

from core.help_bot_logic import FileEditorAgent                                                                                                                                                                

class FileEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI File Editor Agent")
        self.root.geometry("1000x700")
        
        # Initialize agent
        self.agent = FileEditorAgent()
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        # Main panes
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left pane (file browser and editor)
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=2)
        
        # Right pane (AI assistant)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        
        # Folder selection
        folder_frame = ttk.Frame(left_frame, padding=5)
        folder_frame.pack(fill=tk.X)
        
        ttk.Label(folder_frame, text="Folder:").pack(side=tk.LEFT)
        self.folder_label = ttk.Label(folder_frame, text="No folder selected", width=40)
        self.folder_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            folder_frame, 
            text="Browse", 
            command=self.select_folder
        ).pack(side=tk.RIGHT)
        
        # File list
        file_frame = ttk.LabelFrame(left_frame, text="Files", padding=5)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.file_tree = ttk.Treeview(file_frame, columns=("size",), show="tree", height=10)
        self.file_tree.heading("#0", text="Name")
        self.file_tree.heading("size", text="Size")
        self.file_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        self.file_tree.bind("<<TreeviewSelect>>", self.on_file_select)
        
        # Editor
        editor_frame = ttk.LabelFrame(left_frame, text="Editor", padding=5)
        editor_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD, height=15)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.editor.config(state=tk.DISABLED)
        
        # Buttons
        button_frame = ttk.Frame(left_frame, padding=5)
        button_frame.pack(fill=tk.X)
        
        self.save_btn = ttk.Button(
            button_frame,
            text="Save Changes",
            command=self.save_file,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        
        # AI Assistant Panel
        ai_frame = ttk.LabelFrame(right_frame, text="AI Assistant", padding=10)
        ai_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(ai_frame, text="Ask about your file:").pack(anchor=tk.W)
        
        self.user_input = ttk.Entry(ai_frame)
        self.user_input.pack(fill=tk.X, pady=5)
        self.user_input.bind("<Return>", self.ask_ai)
        
        ttk.Button(
            ai_frame, 
            text="Ask AI", 
            command=self.ask_ai
        ).pack(anchor=tk.E, pady=5)
        
        self.ai_response = scrolledtext.ScrolledText(
            ai_frame, 
            wrap=tk.WORD, 
            height=20,
            state=tk.DISABLED
        )
        self.ai_response.pack(fill=tk.BOTH, expand=True)
        
        # Apply AI Suggestion Button
        self.apply_btn = ttk.Button(
            ai_frame,
            text="Apply Suggestion",
            command=self.apply_ai_suggestion,
            state=tk.DISABLED
        )
        self.apply_btn.pack(anchor=tk.E, pady=5)
    
    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.agent.current_folder = folder_selected
            self.folder_label.config(text=folder_selected)
            self.refresh_file_list()
    
    def refresh_file_list(self):
        self.file_tree.delete(*self.file_tree.get_children())
        files = self.agent.load_file_list()
        for name, size in files:
            self.file_tree.insert("", "end", text=name, values=(size,))
    
    def on_file_select(self, event):
        selected = self.file_tree.focus()
        if not selected:
            return
        
        filename = self.file_tree.item(selected, "text")
        try:
            content = self.agent.load_file_content(filename)
            self.editor.config(state=tk.NORMAL)
            self.editor.delete(1.0, tk.END)
            self.editor.insert(tk.END, content)
            self.save_btn.config(state=tk.NORMAL)
            self.agent.ai_suggestion = None
            self.apply_btn.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def save_file(self):
        new_content = self.editor.get(1.0, tk.END)
        try:
            if self.agent.save_file(new_content):
                messagebox.showinfo("Success", "File saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def ask_ai(self, event=None):
        if not self.agent.current_file:
            messagebox.showwarning("No File", "Please select a file first")
            return
            
        user_query = self.user_input.get()
        if not user_query:
            return
            
        # Get current file content from editor
        current_content = self.editor.get(1.0, tk.END)
        filename = os.path.basename(self.agent.current_file)
        
        # Call agent to process AI request
        display_text, modified_content = self.agent.ask_ai(current_content, filename, user_query)
        
        if display_text is None:
            messagebox.showerror("Error", modified_content)
            return
            
        # Display AI response
        self.ai_response.config(state=tk.NORMAL)
        self.ai_response.delete(1.0, tk.END)
        self.ai_response.insert(tk.END, display_text)
        self.ai_response.config(state=tk.DISABLED)
        
        # Enable apply button if we have modified content
        if modified_content:
            self.apply_btn.config(state=tk.NORMAL)
        else:
            self.apply_btn.config(state=tk.DISABLED)
    
    def apply_ai_suggestion(self):
        suggestion = self.agent.apply_ai_suggestion()
        if suggestion:
            self.editor.delete(1.0, tk.END)
            self.editor.insert(tk.END, suggestion)
            messagebox.showinfo("Suggestion Applied", "AI suggestion applied to editor. Click 'Save Changes' to save to file.")
        else:
            # Fallback to text extraction method
            ai_text = self.ai_response.get(1.0, tk.END)
            matches = re.findall(r'```(?:[^\n]*\n)?(.*?)```', ai_text, re.DOTALL)
            
            if matches:
                new_content = matches[-1].strip()
                self.editor.delete(1.0, tk.END)
                self.editor.insert(tk.END, new_content)
                messagebox.showinfo("Suggestion Applied", "AI suggestion applied to editor.")
            else:
                messagebox.showinfo("No Suggestion", "No suggestion found in AI response")

if __name__ == "__main__":
    root = tk.Tk()
    app = FileEditorApp(root)
    root.mainloop()
# from PySide6.QtCore import Qt,Signal
# from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLineEdit, QPushButton, QHBoxLayout, QListWidgetItem
# import webbrowser
# from contactWindowInterface import ContactWindowInterface

# class ContactWindow(QDialog):
#     recommendation_clicked = Signal(dict)
#     def __init__(self, app_name, is_installed, app_config, parent=None):
#         super().__init__(parent)
#         self.app_name = app_name
#         self.is_installed = is_installed
#         self.app_config = app_config
#         self.ui = ContactWindowInterface()
#         self.ui.setupUi(self)

#          # Connect signals
#         self.ui.message_btn.clicked.connect(self._start_chat)
#         self.ui.call_btn.clicked.connect(self._start_call)
#         self.ui.search_bar.textChanged.connect(self._filter_contacts)
        
#         # Initialize
#         self._load_contacts()
#         self._update_status()

#     def setup_ui(self):
#         self.setWindowTitle(f"{self.app_name} - Recent Contacts")
#         self.setFixedSize(450, 550)
        
#         layout = QVBoxLayout()
        
#         # Search bar
#         self.search_bar = QLineEdit()
#         self.search_bar.setPlaceholderText("Search contacts...")
#         self.search_bar.textChanged.connect(self.filter_contacts)
#         layout.addWidget(self.search_bar)
        
#         # Contact list
#         self.contact_list = QListWidget()
#         self.contact_list.itemDoubleClicked.connect(self.start_chat)
#         layout.addWidget(self.contact_list)
        
#         # Action buttons
#         button_layout = QHBoxLayout()
        
#         self.chat_btn = QPushButton("Message")
#         self.chat_btn.clicked.connect(self.start_chat)
#         button_layout.addWidget(self.chat_btn)
        
#         self.call_btn = QPushButton("Call")
#         self.call_btn.clicked.connect(self.start_call)
#         button_layout.addWidget(self.call_btn)
        
#         layout.addLayout(button_layout)
#         self.setLayout(layout)

#     def load_recent_contacts(self):
#         self.contacts = self.get_recent_contacts()
#         self.contact_list.clear()
#         for contact in sorted(self.contacts, key=lambda x: x["last_contact"]):
#             item = QListWidgetItem(f"{contact['name']}\nLast contact: {contact['last_contact']}")
#             item.setData(Qt.UserRole, contact)
#             self.contact_list.addItem(item)

#     def get_recent_contacts(self):
#         return [
#             {"name": "John Doe", "phone": "+1234567890", "id": "john123", "last_contact": "2h ago"},
#             {"name": "Jane Smith", "phone": "+1987654321", "id": "jane456", "last_contact": "Yesterday"},
#             {"name": "Work Group", "phone": "", "id": "team789", "last_contact": "Today"},
#         ]

#     def filter_contacts(self):
#         search_text = self.search_bar.text().lower()
#         for i in range(self.contact_list.count()):
#             item = self.contact_list.item(i)
#             contact = item.data(Qt.UserRole)
#             match = (search_text in contact["name"].lower() or 
#                     search_text in contact.get("phone", "").lower() or
#                     search_text in contact.get("id", "").lower())
#             item.setHidden(not match)

#     def start_chat(self):
#         self.initiate_communication("chat")

#     def start_call(self):
#         self.initiate_communication("call")

#     def initiate_communication(self, action_type):
#         selected = self.contact_list.currentItem()
#         if not selected:
#             return
            
#         contact = selected.data(Qt.UserRole)
#         try:
#             if self.is_installed:
#                 url = self.app_config["deep_links"][action_type].format(
#                     phone=contact.get("phone", ""),
#                     id=contact.get("id", "")
#                 )
#             else:
#                 url = self.app_config["web_urls"][action_type].format(
#                     phone=contact.get("phone", ""),
#                     id=contact.get("id", "")
#                 )
            
#             webbrowser.open(url)
#             self.close()
#         except Exception as e:
#             print(f"Error initiating {action_type}: {e}")
        
#     def closeEvent(self, event):
#             """Clean up when window closes."""
#             self.recommendation_clicked.emit({
#                 "action": "window_closed",
#                 "app": self.app_name
#             })
#             super().closeEvent(event)

from PySide6.QtWidgets import QMainWindow, QMessageBox, QListWidgetItem
from PySide6.QtCore import Signal, Qt
from utils.contactWindowInterface import ContactWindowInterface

class ContactWindow(QMainWindow):
    contact_selected = Signal(dict)
    
    def __init__(self, app_name, is_installed, app_config, contacts, parent=None):
        super().__init__(parent)
        self.app_name = app_name
        self.is_installed = is_installed
        self.app_config = app_config
        
        # Setup UI
        self.ui = ContactWindowInterface()
        self.ui.setupUi(self)
        
        # Connect signals
        self.ui.message_btn.clicked.connect(self._start_chat)
        self.ui.call_btn.clicked.connect(self._start_call)
        self.ui.search_bar.textChanged.connect(self._filter_contacts)
        
        # Initialize
        self._load_contacts(contacts)
        self._update_status()
    
    def _update_status(self):
        status = f"{self.app_name} ({'Installed' if self.is_installed else 'Web'})"
        self.ui.status_label.setText(status)
    
    def _load_contacts(self, contacts):
        """Load actual contacts instead of mock data"""
        self.ui.contact_list.clear()
        for contact in contacts:
            item_text = f"{contact['name']}\nLast: {contact['last_contact']}"
            
            # Add call indicator if available
            if contact.get('has_called', False):
                item_text += " ☎"
                
            # Add phone if available
            if contact.get('phone'):
                item_text += f"\nPhone: {contact['phone']}"
                
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, contact)
            self.ui.contact_list.addItem(item)
    
    def _filter_contacts(self):
        search_text = self.ui.search_bar.text().lower()
        for i in range(self.ui.contact_list.count()):
            item = self.ui.contact_list.item(i)
            contact = item.data(Qt.UserRole)
            match = (
                search_text in contact["name"].lower() or
                search_text in contact.get("phone", "").lower() or
                search_text in contact.get("id", "").lower()
            )
            item.setHidden(not match)
    
    def _start_chat(self):
        self._initiate_action("chat")
    
    def _start_call(self):
        self._initiate_action("call")
    
    def _initiate_action(self, action_type):
        selected = self.ui.contact_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a contact first.")
            return
        
        contact_data = selected.data(Qt.UserRole)
        
        self.contact_selected.emit({
            "action": action_type,
            "contact": contact_data,
            "app_config": self.app_config,
            "is_installed": self.is_installed,
            "app_name": self.app_name
        })
        print("contact Triggered: ", {action_type})
        self.close()
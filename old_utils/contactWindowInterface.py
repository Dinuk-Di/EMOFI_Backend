from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, 
                              QListWidgetItem, QLineEdit, QPushButton, 
                              QVBoxLayout, QWidget, QGroupBox)

class ContactWindowInterface:
    def setupUi(self, ContactWindow):
        if not ContactWindow.objectName():
            ContactWindow.setObjectName(u"ContactWindow")
        ContactWindow.resize(500, 600)
        ContactWindow.setMinimumSize(QSize(500, 500))
        
        # Central Widget
        self.centralwidget = QWidget(ContactWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        
        # Main Layout
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Status Bar
        self.status_group = QGroupBox(self.centralwidget)
        self.status_group.setObjectName(u"StatusGroup")
        self.status_group.setMaximumHeight(50)
        
        status_layout = QHBoxLayout(self.status_group)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.status_group)
        
        # Search Bar
        self.search_container = QFrame()
        search_layout = QHBoxLayout(self.search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search contacts...")
        self.search_bar.setClearButtonEnabled(True)
        self.main_layout.addWidget(self.search_bar)
        self.main_layout.addWidget(self.search_container)
        
        # Contact List with custom items
        self.contact_list = QListWidget()
        self.contact_list.setSpacing(1)  # Add spacing between items
        self.main_layout.addWidget(self.contact_list)

        # Action Buttons with better styling
        self.button_container = QFrame()
        button_layout = QHBoxLayout(self.button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        self.message_btn = QPushButton(" Message ")
        self.call_btn = QPushButton(" Call ")
        
        # Add icons to buttons (replace with your actual icons)
        self.message_btn.setIcon(QIcon("utils/res/Message.png"))
        self.call_btn.setIcon(QIcon("utils/res/Call.png"))
        
        button_layout.addWidget(self.message_btn)
        button_layout.addWidget(self.call_btn)
        
        self.main_layout.addWidget(self.button_container)
        
        # Apply styles
        self._apply_styles()
        
        ContactWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(ContactWindow)
    
    def _apply_styles(self):
        base_style = """
            QGroupBox {
                background: rgba(239, 207, 207, 180);
                border-radius: 10px;
                border: 1px solid rgba(0, 180, 255, 150);
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLineEdit {
                padding: 8px;
                border-radius: 10px;
                border: 1px solid #ccc;
                font-size: 14px;
                background: white;
            }
            QListWidget {
                background: white;
                border-radius: 10px;
                border: 1px solid #ddd;
                font-size: 14px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            #MessageBtn {
                background-color: #4CAF50;
                color: white;
            }
            #MessageBtn:hover {
                background-color: #45a049;
            }
            #CallBtn {
                background-color: #2196F3;
                color: white;
            }
            #CallBtn:hover {
                background-color: #0b7dda;
            }
        """
        self.centralwidget.setStyleSheet(base_style)
        self.message_btn.setObjectName("MessageBtn")
        self.call_btn.setObjectName("CallBtn")
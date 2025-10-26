# -*- coding: utf-8 -*-

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt, Signal, QPropertyAnimation
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget, QPushButton, QGroupBox)
from PySide6.QtWidgets import QLineEdit, QPushButton

class ClickableFrame(QFrame):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.normal_color = "rgba(239, 224, 225, 128)"
        self.hover_color = "rgba(239, 207, 207, 200)"  # Slightly brighter/lighter
        self.pressed_color = "rgba(239, 207, 207, 255)"  # Darker when pressed
        
        self.setStyleSheet(f"""
            ClickableFrame {{
                background-color: {self.normal_color};
                border-radius: 15px;
                border: 2px solid rgb(0, 255, 255);
            }}
        """)

    def enterEvent(self, event):
        self.setStyleSheet(f"""
            ClickableFrame {{
                background-color: {self.hover_color};
                border-radius: 15px;
                border: 2px solid rgb(0, 255, 255);
            }}
        """)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.setStyleSheet(f"""
            ClickableFrame {{
                background-color: {self.normal_color};
                border-radius: 15px;
                border: 2px solid rgb(0, 255, 255);
            }}
        """)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setStyleSheet(f"""
                ClickableFrame {{
                    background-color: {self.pressed_color};
                    border-radius: 15px;
                    border: 2px solid rgb(0, 200, 200);
                }}
            """)
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setStyleSheet(f"""
            ClickableFrame {{
                background-color: {self.hover_color};
                border-radius: 15px;
                border: 2px solid rgb(0, 255, 255);
            }}
        """)
        super().mouseReleaseEvent(event)

class InteraceMainwindow(object):

    def add_choice(self, text="New Choice", id=None, icon_path=None, on_click=None):
        """Improved version with proper parameter usage"""
        choice_frame = ClickableFrame()
        choice_frame.setObjectName(text.replace(" ", "_"))  # Sanitize object name
        choice_frame.setMinimumSize(QSize(380, 60))
        choice_frame.setMaximumSize(QSize(380, 60))
        
        # Set style with proper selector
        choice_frame.setStyleSheet(f"""
        #{choice_frame.objectName()} {{
           
            border-radius: 15px;
            border: 2px solid rgb(0, 255, 255);
        }}
        #{choice_frame.objectName()}:hover {{
        }}
        #{choice_frame.objectName()}:pressed {{
        }}
        """)
        
        layout = QHBoxLayout(choice_frame)
        layout.setContentsMargins(5, 5, 5, 5)

        # Icon frame
        icon_frame = QFrame()
        icon_frame.setObjectName("ChoiceIcon")
        icon_frame.setFixedSize(QSize(60, 50))
        
        if icon_path:
            icon_frame.setStyleSheet(f"""
            #ChoiceIcon {{
                background-image: url({icon_path});
                background-position: center;
                background-repeat: no-repeat;
                border-radius: 15px;
            }}
            """)
        
        layout.addWidget(icon_frame)

        # Text frame
        text_frame = QFrame()
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        text_label.setFont(QFont("Sans Serif", 11, QFont.Bold))
        
        text_layout = QVBoxLayout(text_frame)
        text_layout.addWidget(text_label)
        layout.addWidget(text_frame)

        # Connect click signal if provided
        if callable(on_click):
            choice_frame.clicked.connect(lambda: on_click(text,id))

        self.verticalLayout.addWidget(choice_frame)

    def setupUi(self, MainWindow):
        # ===== Main Window Setup =====
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(530, 240)
        MainWindow.setMinimumSize(QSize(527, 235))
        MainWindow.setMaximumSize(QSize(530, 240))
        MainWindow.setLayoutDirection(Qt.LeftToRight)

        # ===== Stylesheet =====
        MainWindow.setStyleSheet(u"""
        * {
            border: none;
            background-color: transparent;
            background: none;
            padding: 0;
            margin: 0;
        }                

        #ChoiceFrame {
            background-color: rgba(239, 207, 207, 100);
            border-radius: 15px;
        }

        #Icon {
            background-color: rgba(87, 117, 201, 0.392);
            border-image: url(utils/res/Icon.jpg);
            border-radius: 15px;
        }
                                 
        #Close_Btn {
	        border-radius: 15px;
        }
        """)

        # ===== Central Widget =====
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        # ===== Main Frame =====
        self.MainFrame = QFrame(self.centralwidget)
        self.MainFrame.setObjectName(u"MainFrame")
        self.MainFrame.setGeometry(QRect(9, 9, 511, 221))
        self.MainFrame.setMinimumSize(QSize(511, 221))
        self.MainFrame.setMaximumSize(QSize(511, 221))
        self.MainFrame.setFrameShape(QFrame.StyledPanel)
        self.MainFrame.setFrameShadow(QFrame.Raised)

        # ===== Hide element =====
        self.recommend_group = QGroupBox(self.MainFrame)
        self.recommend_group.setObjectName(u"Recommend_Group")

        # ===== Exit button =====
        self.Close_Btn = QPushButton(self.recommend_group)
        self.Close_Btn.setObjectName(u"Close_Btn")

        self.Close_Btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 30);
                border-radius: 3px;
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 50);
            }
        """)

        self.Close_Btn.setEnabled(True)
        self.Close_Btn.setGeometry(QRect(10, 30, 31, 21))
        self.Close_Btn.setAutoFillBackground(False)
        icon = QIcon()
        icon.addFile(u"utils/res/close.png", QSize(), QIcon.Normal, QIcon.Off)
        self.Close_Btn.setIcon(icon)
        self.Close_Btn.raise_()

        # ===== Icon Frame =====
        self.Icon = QFrame(self.MainFrame)
        self.Icon.setObjectName(u"Icon")
        self.Icon.setGeometry(QRect(410, 0, 100, 100))
        self.Icon.setMinimumSize(QSize(100, 100))
        self.Icon.setBaseSize(QSize(100, 100))
        self.Icon.setFrameShape(QFrame.Box)
        self.Icon.setFrameShadow(QFrame.Raised)

        # ===== Choice Frame =====
        self.ChoiceFrame = QFrame(self.recommend_group)
        self.ChoiceFrame.setObjectName(u"ChoiceFrame")
        self.ChoiceFrame.setGeometry(QRect(10, 49, 421, 162))
        self.ChoiceFrame.setMinimumSize(QSize(421, 161))
        self.ChoiceFrame.setLayoutDirection(Qt.LeftToRight)
        self.ChoiceFrame.setFrameShape(QFrame.StyledPanel)
        self.ChoiceFrame.setFrameShadow(QFrame.Raised)

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.ChoiceFrame.setSizePolicy(sizePolicy)

        self.verticalLayout = QVBoxLayout(self.ChoiceFrame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSpacing(9)
        self.verticalLayout.setContentsMargins(15, 18, 26, 15)

        # frame_1 = add_choice("Watch Something New", "youtube_resize.png")
        # self.verticalLayout.addWidget(frame_1)

        # frame_2 = add_choice("Watch Something New", "youtube_resize.png")
        # self.verticalLayout.addWidget(frame_2)

        # Raise elements
        self.ChoiceFrame.raise_()
        self.Icon.raise_()

        self.recommend_group.setVisible(True)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

        
        # Add search components (hidden by default)
        self.search_frame = QFrame(self.MainFrame)
        self.search_frame.setObjectName(u"SearchFrame")
        self.search_frame.setGeometry(QRect(10, 49, 421, 100))
        self.search_frame.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(239, 207, 207, 180),
                    stop:1 rgba(200, 230, 255, 180)
                );
                border-radius: 15px;
                border: 2px solid rgb(0, 180, 255);
            }
        """)
        self.search_frame.hide()
        
        self.search_input = QLineEdit(self.search_frame)
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_input.setGeometry(QRect(20, 20, 300, 30))
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 10px;
                border: 1px solid rgb(0, 150, 255);
                padding: 5px;
                font-size: 12px;
            }
            QLineEdit:hover {
                background-color: rgba(255, 255, 255, 230);
                border: 1px solid rgb(0, 180, 255);
            }
            QLineEdit:focus {
                background-color: rgba(255, 255, 255, 240);
                border: 1px solid rgb(0, 200, 255);
            }
        """)
        
        self.search_button = QPushButton("Search", self.search_frame)
        self.search_button.setGeometry(QRect(330, 20, 70, 30))
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(100, 200, 255, 220),
                    stop:1 rgba(50, 150, 255, 220)
                );
                border-radius: 10px;
                border: 1px solid rgb(0, 150, 255);
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(120, 220, 255, 240),
                    stop:1 rgba(70, 170, 255, 240)
                );
                border: 1px solid rgb(0, 180, 255);
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(80, 180, 255, 240),
                    stop:1 rgba(30, 130, 255, 240)
                );
                border: 1px solid rgb(0, 200, 255);
                padding-top: 2px;
                padding-left: 2px;
            }
        """)
        self.search_button.clicked.connect(self.on_search_clicked)
        
        
        self.cancel_button = QPushButton("Cancel", self.search_frame)
        self.cancel_button.setGeometry(QRect(330, 60, 70, 30))
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 150, 150, 220),
                    stop:1 rgba(255, 100, 100, 220)
                );
                border-radius: 10px;
                border: 1px solid rgb(255, 100, 100);
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 170, 170, 240),
                    stop:1 rgba(255, 120, 120, 240)
                );
                border: 1px solid rgb(255, 120, 120);
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 130, 130, 240),
                    stop:1 rgba(255, 80, 80, 240)
                );
                border: 1px solid rgb(255, 140, 140);
                padding-top: 2px;
                padding-left: 2px;
            }
        """)


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))

    def updateLayoutShow(self, Recommend_Group):
        self.recommend_group.setVisible(True)
    
    def show_search(self, search_type="youtube"):
        """Show search frame with type-specific settings"""
        self.search_frame.setVisible(True)
        self.ChoiceFrame.setVisible(False)
        
        if search_type == "telegram":
            self.search_input.setPlaceholderText("Enter Telegram username (e.g., @username)")
            self.search_button.setText("Find User")
        else:  # youtube
            self.search_input.setPlaceholderText("Enter search query...")
            self.search_button.setText("Search")

    def on_search_clicked(self):
        search_query = self.search_input.text()
        # Emit signal or handle the search query
        # If no query was entered, use the default from the selected choice
        if not search_query:
            if hasattr(self, 'selected_choice'):
                search_query = self.selected_choice.get('search_query', '')
        
        # Determine which callback to use based on placeholder text
        if "Telegram username" in self.search_input.placeholderText():
            if hasattr(self, 'telegram_search_callback'):
                self.telegram_search_callback(search_query)
        else:
            if hasattr(self, 'search_callback'):
                self.search_callback(search_query)
        self.show_search(False)

    def on_cancel_clicked(self):
        self.show_search(False)
        if hasattr(self, 'search_callback'):
            self.search_callback(None)

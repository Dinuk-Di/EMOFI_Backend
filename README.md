# Facial Emotion Recognition Desktop App

A modern desktop application for real-time facial emotion recognition using computer vision and machine learning.

## 🚀 Quick Start

### For End Users:
1. Download `FacialEmotionRecognition_Portable.zip`
2. Extract to any folder
3. Run `FacialEmotionRecognition.exe`

### For Developers:
```bash
# Clone the repository
git clone <your-repo-url>
cd DesktopApp

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 📋 Features

- **Real-time Emotion Detection**: Analyze facial expressions in real-time
- **User Authentication**: Secure login and registration system
- **Hand Movement Tracking**: Detect and track hand gestures
- **Sleepy Detection**: Monitor for signs of drowsiness
- **Human Detection**: Identify human presence in camera feed
- **Modern UI**: Beautiful CustomTkinter interface
- **Database Integration**: SQLite database for user management
- **Agent System**: AI-powered assistance features

## 🛠️ System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 100MB free space
- **Camera**: Webcam for emotion detection
- **Python**: 3.8+ (for development)

## 📦 Installation

### Option 1: Portable Version (Recommended)
1. Download `FacialEmotionRecognition_Portable.zip`
2. Extract to any location
3. Run `FacialEmotionRecognition.exe`

### Option 2: Installer
1. Download `FacialEmotionRecognition_Setup.exe`
2. Run as Administrator
3. Follow installation wizard
4. Launch from Start Menu

### Option 3: From Source
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## 🔧 Building from Source

### Prerequisites:
- Python 3.8+
- PyInstaller
- NSIS (optional, for installer creation)

### Build Commands:
```bash
# Quick build
build.bat

# Or use PowerShell
powershell -ExecutionPolicy Bypass -File build.ps1

# Or manual build
python build_installer.py
```

## 📁 Project Structure

```
DesktopApp/
├── main.py                 # Application entry point
├── ui/                     # User interface modules
│   ├── login.py           # Login window
│   ├── register.py        # Registration window
│   └── dashboard.py       # Main dashboard
├── core/                   # Core functionality
│   ├── emotion_detector.py # Emotion recognition
│   ├── hand_movement.py   # Hand tracking
│   ├── sleepy_detector.py # Drowsiness detection
│   ├── human_detector.py  # Human detection
│   ├── agent_system.py    # AI agent system
│   └── controller.py      # Main controller
├── Models/                 # Machine learning models
├── assets/                 # Application assets
├── database/               # Database modules
└── utils/                  # Utility modules
```

## 🎯 Usage

1. **Launch Application**: Run the executable or use `python main.py`
2. **Login/Register**: Create an account or login with existing credentials
3. **Access Dashboard**: Use the main interface to access features
4. **Enable Camera**: Grant camera permissions for emotion detection
5. **Start Detection**: Click on emotion detection features to begin analysis

## 🔒 Privacy & Security

- All data is stored locally on your device
- No data is transmitted to external servers
- Camera access is only used for emotion detection
- User credentials are securely stored in local database

## 🐛 Troubleshooting

### Common Issues:

1. **Application won't start**
   - Ensure you have the latest version
   - Check system requirements
   - Run as Administrator if needed

2. **Camera not working**
   - Check camera permissions
   - Ensure no other application is using the camera
   - Try restarting the application

3. **Missing dependencies**
   - For portable version: Extract all files
   - For source: Run `pip install -r requirements.txt`

### Debug Mode:
```bash
# For developers: Enable console output
# Edit main.spec: change console=False to console=True
pyinstaller main.spec --clean
```

## 📞 Support

- **Documentation**: See `INSTALLATION_GUIDE.md` for detailed instructions
- **Issues**: Check `BUILD_SUMMARY.md` for troubleshooting
- **Development**: Review source code in respective modules

## 📄 License

This project is licensed under the MIT License - see `LICENSE.txt` for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📈 Roadmap

- [ ] Enhanced emotion accuracy
- [ ] Additional gesture recognition
- [ ] Cloud synchronization (optional)
- [ ] Mobile app companion
- [ ] API integration
- [ ] Advanced analytics dashboard

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Maintainer**: Your Name/Organization 
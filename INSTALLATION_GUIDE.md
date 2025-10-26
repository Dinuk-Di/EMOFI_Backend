# Facial Emotion Recognition - Installation Guide

## Overview
This guide provides step-by-step instructions for creating an installable setup for the Facial Emotion Recognition desktop application.

## Prerequisites

### For Development/Building:
1. **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
2. **Git** - [Download Git](https://git-scm.com/downloads)
3. **NSIS (Optional)** - [Download NSIS](https://nsis.sourceforge.io/Download) for creating installers

### For End Users:
- Windows 10/11 (64-bit)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Webcam for emotion detection features

## Quick Start (For End Users)

### Option 1: Installer (Recommended)
1. Download `FacialEmotionRecognition_Setup.exe`
2. Run the installer as Administrator
3. Follow the installation wizard
4. Launch from Start Menu or Desktop shortcut

### Option 2: Portable Version
1. Download `FacialEmotionRecognition_Portable.zip`
2. Extract to any folder
3. Run `FacialEmotionRecognition.exe`

## Building the Application

### Step 1: Setup Development Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd DesktopApp

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -r build_requirements.txt
```

### Step 2: Test the Application
```bash
# Run the application
python main.py
```

### Step 3: Build Executable
```bash
# Run the automated build script
python build_installer.py
```

This will create:
- `dist/FacialEmotionRecognition/FacialEmotionRecognition.exe` - Standalone executable
- `FacialEmotionRecognition_Setup.exe` - Windows installer (if NSIS is installed)
- `FacialEmotionRecognition_Portable.zip` - Portable version

### Step 4: Manual Build (Alternative)
```bash
# Build with PyInstaller
pyinstaller main.spec --clean

# Create installer (if NSIS is installed)
"C:\Program Files\NSIS\makensis.exe" installer.nsi
```

## File Structure

```
DesktopApp/
├── main.py                 # Application entry point
├── main.spec              # PyInstaller specification
├── installer.nsi          # NSIS installer script
├── build_installer.py     # Automated build script
├── requirements.txt       # Application dependencies
├── build_requirements.txt # Build dependencies
├── LICENSE.txt           # License file
├── ui/                   # User interface modules
├── core/                 # Core functionality
├── Models/               # ML models
├── assets/               # Application assets
├── database/             # Database modules
└── utils/                # Utility modules
```

## Troubleshooting

### Common Issues:

1. **"Missing module" errors during build**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that all imports are properly included in `main.spec`

2. **Large executable size**
   - This is normal for ML applications with TensorFlow/PyTorch
   - Consider using UPX compression (already enabled in spec file)

3. **NSIS installer fails**
   - Ensure NSIS is properly installed
   - Run as Administrator
   - Check that all paths in `installer.nsi` are correct

4. **Application crashes on startup**
   - Check if all model files are present in `Models/` directory
   - Ensure database files exist in `assets/` directory
   - Run with console enabled for debugging: change `console=False` to `console=True` in `main.spec`

### Debug Mode:
To run the application with console output for debugging:
```bash
# Modify main.spec: change console=False to console=True
# Then rebuild
pyinstaller main.spec --clean
```

## Customization

### Changing Application Name:
1. Update `APP_NAME` in `installer.nsi`
2. Update `name` in `main.spec`
3. Update `build_installer.py` references

### Adding Icons:
1. Place your icon file in `assets/icon.ico`
2. Update icon paths in `main.spec` and `installer.nsi`

### Modifying Installer:
- Edit `installer.nsi` for custom installer behavior
- Add additional sections for optional components
- Customize branding and appearance

## Distribution

### Recommended Files to Distribute:
1. `FacialEmotionRecognition_Setup.exe` - For most users
2. `FacialEmotionRecognition_Portable.zip` - For portable use
3. `README.md` - User documentation
4. `CHANGELOG.md` - Version history

### File Sizes:
- Executable: ~200-500MB (typical for ML applications)
- Installer: ~150-400MB (compressed)
- Portable: ~200-500MB

## Security Considerations

1. **Code Signing**: Consider code signing your executable for better Windows compatibility
2. **Antivirus**: Some antivirus software may flag ML applications - consider whitelisting
3. **Permissions**: The application requires camera access for emotion detection

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review application logs
3. Test with console enabled for detailed error messages
4. Ensure all dependencies and models are properly installed

## Version History

- v1.0.0 - Initial release with basic installer support
- Future versions will include additional features and improvements 
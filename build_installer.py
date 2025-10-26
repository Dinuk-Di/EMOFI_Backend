#!/usr/bin/env python3
"""
Build script for Facial Emotion Recognition Desktop App
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 Running: {description}")
    print(f"📌 Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")

    # PyInstaller
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
        print("✅ PyInstaller found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyInstaller not found. Installing...")
        run_command("pip install pyinstaller", "Installing PyInstaller")

    # NSIS (optional)
    nsis_path = None
    possible_paths = [
        r"C:\Program Files\NSIS\makensis.exe",
        r"C:\Program Files (x86)\NSIS\makensis.exe",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            nsis_path = path
            break

    if nsis_path:
        print("✅ NSIS found")
    else:
        print("⚠️  NSIS not found. Installer will be skipped.")

    return nsis_path

def clean_build():
    """Clean previous build artifacts"""
    print("\n🧹 Cleaning previous build artifacts...")
    for folder in ['build', 'dist', '__pycache__']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🗑️  Removed {folder}")

    for file in os.listdir('.'):
        if file.endswith('.spec') and file != 'main.spec':
            os.remove(file)
            print(f"🗑️  Removed {file}")

def build_executable():
    """Build the executable using PyInstaller"""
    print("\n⚙️ Building executable...")
    success = run_command(
        "pyinstaller main.spec --clean",
        "PyInstaller Build"
    )
    exe_path = "dist/FacialEmotionRecognition/FacialEmotionRecognition.exe"
    if success and os.path.exists(exe_path):
        print(f"✅ Executable created: {exe_path}")
        return True
    else:
        print("❌ Executable not found.")
        return False

def create_installer(nsis_path):
    """Create installer using NSIS"""
    if not nsis_path:
        print("⚠️  Skipping installer creation (NSIS not found)")
        return False

    print("\n📦 Creating installer using NSIS...")
    success = run_command(
        f'"{nsis_path}" installer.nsi',
        "NSIS Installer"
    )

    if success and os.path.exists("FacialEmotionRecognition_Setup.exe"):
        size = os.path.getsize("FacialEmotionRecognition_Setup.exe") / (1024 * 1024)
        print(f"✅ Installer created: FacialEmotionRecognition_Setup.exe ({size:.1f} MB)")
        return True
    else:
        print("❌ Installer creation failed.")
        return False

def create_portable_zip():
    """Create a portable ZIP version"""
    print("\n📦 Creating portable ZIP...")
    if not os.path.exists("dist/FacialEmotionRecognition"):
        print("❌ Build folder not found!")
        return False

    zip_path = "FacialEmotionRecognition_Portable.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    success = run_command(
        'powershell Compress-Archive -Path "dist/FacialEmotionRecognition/*" -DestinationPath "FacialEmotionRecognition_Portable.zip"',
        "Create Portable ZIP"
    )

    if success and os.path.exists(zip_path):
        size = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"✅ ZIP created: {zip_path} ({size:.1f} MB)")
        return True
    else:
        print("❌ ZIP creation failed.")
        return False

def main():
    """Main build process"""
    print("🚀 Facial Emotion Recognition - Build Script")
    print("=" * 50)

    nsis_path = check_dependencies()
    clean_build()

    if not build_executable():
        sys.exit(1)

    installer = create_installer(nsis_path)
    portable = create_portable_zip()

    print("\n" + "=" * 50)
    print("📋 BUILD SUMMARY")
    print("=" * 50)

    if os.path.exists("dist/FacialEmotionRecognition/FacialEmotionRecognition.exe"):
        print("✅ Executable: dist/FacialEmotionRecognition/FacialEmotionRecognition.exe")
    if installer:
        print("✅ Installer: FacialEmotionRecognition_Setup.exe")
    if portable:
        print("✅ Portable: FacialEmotionRecognition_Portable.zip")

    if not installer:
        print("\n💡 Tip: Install NSIS to build an installer: https://nsis.sourceforge.io/Download")

if __name__ == "__main__":
    main()

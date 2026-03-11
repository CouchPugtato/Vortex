@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0connect_jetson.ps1" -Alias jetson-usb -ReplaceKey %*

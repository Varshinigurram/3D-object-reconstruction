
import sys
import cv2
import numpy as np
import open3d as o3d
import streamlit as st
import plotly
import pandas as pd

print("=" * 45)
print("COMPUTER VISION ENVIRONMENT CHECK")
print("=" * 45)

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Open3D:", o3d.__version__)
print("Streamlit:", st.__version__)
print("Plotly:", plotly.__version__)
print("Pandas:", pd.__version__)

# Verify SIFT availability
sift = cv2.SIFT_create()

print("\nSIFT available: YES")

# Verify Open3D geometry creation
point_cloud = o3d.geometry.PointCloud()

print("Open3D geometry: WORKING")

print("\nENVIRONMENT CHECK COMPLETED")

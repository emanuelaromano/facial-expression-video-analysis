# HireView Prep

A comprehensive interview preparation platform that analyzes video recordings to provide feedback on facial expressions, emotions, and speech patterns. Built with React frontend and FastAPI backend, featuring AI-powered analysis using MediaPipe, OpenAI, and emotion detection models.

## 🚀 Features

- **Video Upload & Processing**: Upload interview videos for AI analysis
- **Facial Expression Analysis**: Real-time facial landmark detection using MediaPipe
- **Emotion Recognition**: AI-powered emotion classification throughout the video
- **Speech Analysis**: OpenAI-powered transcript generation and analysis
- **Interactive Dashboard**: Beautiful UI with charts and statistics
- **Authentication System**: JWT-based user authentication
- **Real-time Processing**: Background task processing with status updates
- **Cloud Deployment**: Ready for Google Cloud Run and Firebase hosting

## 🎥 Demo

<video width="100%" controls>
  <source src="demo.mp4" type="video/quicktime">
  Your browser does not support the video tag.
</video>

## 🏗️ Architecture

### Frontend (React + Vite)
- **React 19** with modern hooks and functional components
- **Redux Toolkit** for state management
- **Tailwind CSS + DaisyUI** for styling
- **React Router** for navigation
- **Plotly.js** for data visualization
- **FFmpeg.wasm** for client-side video processing

### Backend (FastAPI + Python)
- **FastAPI** for high-performance API
- **MediaPipe** for facial landmark detection
- **OpenAI API** for speech-to-text and analysis
- **Transformers** for emotion classification
- **Redis** for real-time status updates
- **Google Cloud Storage** for file storage
- **Docker** containerization
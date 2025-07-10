# Image Transformation and Generation using AI

## Project Overview:
This project is a web-based AI platform that offers advanced image transformation and generation features using machine learning APIs. Users can upload images or prompts to apply style transfer, super-resolution, colorization, background removal, and text-to-image generation.

## 1. Authentication Pages
### Signup Page
Fields: Name, Email, Password, Confirm Password

Features: Password match and strength check

Redirects to login after successful signup

### Login Page
Fields: Email, Password

Features: Validation (correct format, empty fields)

Redirects to ML Features Page upon success

## 2. ML Features Dashboard (After Login)
### Layout
Card-style UI listing all available ML features

Clicking on a module redirects to its dedicated page

## 🎨2.1 Colorization
Purpose: Convert black and white photos into color using AI.

Input: Upload grayscale image

Backend: Uses DeOldify or similar models via Replicate API.

Output: Colorized version of the image

UI Flow: File upload → “Colorize” button → Loading → Display Result + Download Option

## 🌑 2.2 Decolorization
Purpose: Convert color images to realistic grayscale

Input: Upload a color image

Backend: Use OpenCV or PIL to convert image to grayscale on the server

Output: Grayscale version of the image

UI Flow: Upload → “Decolorize” → View result

## 🖼️ 2.3 Text-to-Image Generation
Purpose: Create AI-generated images from text prompts

Input: Text prompt (e.g., “a futuristic city at night”)

Backend: Uses Stable Diffusion or DALL·E via Replicate API

Output: Image generated from the prompt

UI Flow: Enter prompt → “Generate Image” → Show generated image → Option to regenerate or download

## 🧾 2.4 Image-to-Text Generation
Purpose: Generate a meaningful AI-based caption describing the contents of an image.

Input: Upload an image (JPG, PNG)

Backend: Uses a pre-trained image captioning model integrated via Replicate API  or HuggingFace Inference API.

Output: A natural language caption like “A man riding a bicycle down a city street”

UI Flow: Upload → Click “Generate Caption” → Backend returns caption → Display the result with options to regenerate or copy




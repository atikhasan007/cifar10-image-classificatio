# cifar10-image-classificatio


# 🐔 CIFAR-10 Image Classification Web App (Flask + CNN)

## 📌 Project Overview

This project is a deep learning-based image classification system built using a Convolutional Neural Network (CNN) and deployed with a Flask web application. The model is trained on the CIFAR-10 dataset, which contains 10 different classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

The application allows users to upload an image through a web interface, processes the image using the trained model, and returns the predicted class along with a confidence score.

---

## 🎯 Project Objective

The main objective of this project is to build an end-to-end image classification system that can accurately classify images into CIFAR-10 categories using deep learning techniques.

### Key Objectives:
- Build and train a CNN model for image classification  
- Use CIFAR-10 dataset for training and evaluation  
- Develop a Flask-based web application for real-time prediction  
- Implement image preprocessing and normalization  
- Deploy the trained model for user interaction  
- Evaluate model performance using accuracy and loss metrics  

---

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Flask  
- HTML, CSS, JavaScript  
- PIL / OpenCV  

---

## 📊 Dataset

- CIFAR-10 Dataset  
- 10 Classes:
  - Airplane  
  - Automobile  
  - Bird  
  - Cat  
  - Deer  
  - Dog  
  - Frog  
  - Horse  
  - Ship  
  - Truck  

---





## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/atikhasan007/cifar10-image-classification.git
cd cifar10-image-classification
```

### setup anaconda  environment and Install dependencies
```base
conda create -n abc python=3.8
conda activate abc
pip install -r requiremetns.txt
```

### ⚙️ Project Workflow
```bash
Update config.yaml
Update params.yaml
Update the entity
Update the configuration manager in src config
Update the components
Update the pipeline
Update the main.py
Update the app.py   

```

# USER UI 

<img width="686" height="728" alt="Screenshot 2026-07-01 193025" src="https://github.com/user-attachments/assets/9483cb75-9fbf-4c67-9e9e-203ffd0b7ef5" />
<img width="967" height="997" alt="Screenshot 2026-07-01 192927" src="https://github.com/user-attachments/assets/18535123-f5cc-4eae-bb20-5d2178cf6abd" />
<img width="915" height="992" alt="Screenshot 2026-07-01 192832" src="https://github.com/user-attachments/assets/7f9cb3dc-2a8f-41a9-bf1f-abbc2076c239" />
<img width="927" height="1002" alt="Screenshot 2026-07-01 192733" src="https://github.com/user-attachments/assets/7c91d444-5076-4953-9132-fe43873507d2" />
<img width="925" height="990" alt="Screenshot 2026-07-01 192613" src="https://github.com/user-attachments/assets/d0c9619d-bb00-44db-a4a9-c7fac470d39d" />
<img width="938" height="1017" alt="Screenshot 2026-07-01 192548" src="https://github.com/user-attachments/assets/29db3893-506e-40ee-aae6-06b64a3ec297" />
<img width="940" height="866" alt="Screenshot 2026-07-01 192300" src="https://github.com/user-attachments/assets/1b16b712-96ee-468b-ae6d-c4ec94530092" />

## Challenges Faced During Development

### GitHub Repository Management
- 1. GitHub Repository Management
- Faced difficulties pushing the project to GitHub after the data ingestion stage because of the large dataset size.
  
- Multiple attempts to push the repository were unsuccessful due to Git limitations and repository size.
- To resolve the issue, the Git repository was cleaned, reinitialized, and configured with an appropriate .gitignore file before pushing the project again.
  
- As a result of reinitializing the Git repositor y, the previous commit history was lost, reducing the total number of Git commits.

### Hardware Limitations
- 2.Hardware Limitations
- The development machine did not have a dedicated GPU for deep learning model training.
- Therefore, the model was trained entirely on the CPU, which significantly increased the training time.
- Despite the hardware limitation, the complete end-to-end pipeline was successfully implemented and tested successfully.

### Debugging and Pipeline Integration
- 3.Debugging and Pipeline Integration
- Encountered several configuration and implementation issues while developing the modular end-to-end pipeline.
- Debugged problems related to project structure, configuration files, training pipeline, dependency management, and module integration.
- Successfully resolved the issues to ensure a stable, reproducible, and fully functional end-to-end workflow.



## Future Improvements

### Infrastructure & Deployment
- Docker containerization for consistent deployment
- AWS/GCP based cloud deployment
- Model serving and scalable API deployment
- Distributed data storage and caching

### DevOps & CI/CD
- GitHub Actions based CI/CD pipeline
- Automated testing and deployment workflow
- Monitoring and logging integration

### Codebase Improvements
- Modular clean architecture
- Service-based project structure
- Improved error handling and validation
- Environment-based configuration management
- Improve model performance while enhancing accuracy.

### UI/UX Improvements
- More interactive and responsive UI
- Dashboard visualization
- Better accessibility and user experience

### System Design & Documentation
- High Level Design (HLD)
- Low Level Design (LLD)
- System architecture diagrams
- Wireframes and workflow diagrams
- Technical documentation and reporting

##   Author & Contact 
- Md Atik Hasan
- Email : imatik513@gmail.com
- Phone : +8801827693853


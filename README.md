## 🏆 **IITB CLASSIFI Hackathon Winning Project**

# 🧠 Machine Learning Evolution: Training the Brain of ClassifyMe.ai

ClassifyMe.ai’s success hinges on a robust, multi-stage machine learning pipeline. This pipeline has allowed the system to not only classify Documents accurately but to also evolve over time, learning from its mistakes and improving with every interaction. Below is the detailed journey of fine-tuning and adapting a pre-trained model to handle the complex task of Document classification.
![WhatsApp Image 2024-12-01 at 21 33 44_2bcf287f](https://github.com/user-attachments/assets/8800d0b3-ec80-4dab-9c18-22d483e66bcd)
![WhatsApp Image 2024-12-01 at 21 34 20_ce3136e5](https://github.com/user-attachments/assets/a140487d-d1db-4299-917a-fea0d9d975c5)
---
![WhatsApp Image 2024-12-05 at 13 48 14_1126fd71](https://github.com/user-attachments/assets/39fbdceb-9bb7-4900-9719-83095589d7b6)

![WhatsApp Image 2024-12-05 at 13 46 57_b0baf676](https://github.com/user-attachments/assets/1e18821b-9358-43a0-aba3-5ec2a8228e86)

![WhatsApp Image 2024-12-05 at 13 45 54_0e53d737](https://github.com/user-attachments/assets/046c275a-961e-486f-83c3-ed93d27fdb1a)
![WhatsApp Image 2024-12-05 at 13 46 01_cd1b5f3b](https://github.com/user-attachments/assets/fd74c61e-607c-4b8b-b77d-f9e4b49af0fa)
![WhatsApp Image 2024-12-05 at 13 46 13_d30ab328](https://github.com/user-attachments/assets/4a597247-f8be-4910-add3-2cad7ee517fd)


![WhatsApp Image 2024-12-01 at 21 35 44_21e83b68](https://github.com/user-attachments/assets/27734ed3-f6ed-4001-8adb-6ff120452c64)

## 🔄 The Fine-Tuning Journey: From Simplicity to Precision

When we began this project, our first goal was to find a model that could understand and process the varying structures of Documents. Early on, we realized that fine-tuning an existing pre-trained model was the key to achieving both accuracy and efficiency. The backbone of our solution was **BERT** 🧑‍💻—a transformer-based language model that has proven to excel at contextual understanding in NLP tasks.

### **Why Fine-Tuning?** 🤔

Fine-tuning refers to the process of taking a pre-trained model (like BERT) and training it further on a specific dataset—in our case, a collection of Documents. The purpose of fine-tuning is to adapt the pre-trained model’s general language understanding to a more specialized task, such as Document classification. 

Fine-tuning allows the model to:
- **Adapt to Specific Domain** 🔧: Document content varies widely from general text. Fine-tuning allows the model to learn domain-specific terms, context, and structure that are unique to Documents.
- **Boost Performance** 📈: Since the base BERT model is already trained on vast amounts of data, fine-tuning on our dataset results in faster learning and higher accuracy.
- **Leverage Pre-Trained Knowledge** 🧠: By starting with a model that has already learned about language and context, fine-tuning ensures that we don't have to start from scratch, saving time and computational resources.

---

## **The Initial Approach: Trying Multiple Models** 🧪

Before landing on BERT, we explored several classification models to see which one could best handle the nuances of Document data.

### **1. Logistic Regression** 💡

We initially tried a simple **Logistic Regression** model, using basic feature extraction methods like **TF-IDF** to represent the Documents. While this model was quick to implement, the results were underwhelming. The accuracy hovered around **65%**, and it struggled to generalize across different types of Documents. The simplicity of logistic regression couldn't capture the complexity and context of Document language.

### **2. Naive Bayes** 🧑‍🏫

Next, we experimented with **Naive Bayes**, another classic model for text classification. Like Logistic Regression, it performed better than random chance but still left much to be desired. With an accuracy of **70%**, it couldn't handle nuances like the relationship between various Document sections (e.g., skills and job roles).

### **3. Random Forests** 🌳

We also tried **Random Forests**, which offered improved accuracy due to the ensemble method’s ability to handle complex features. However, the accuracy was still limited to **75%**, and the model struggled with understanding the hierarchical structure of Documents (e.g., sections like "Education," "Experience," and "Skills").

### **4. Support Vector Machines (SVM)** 💻

After random forests, we tried **SVMs** with a **linear kernel**. While this model performed better than previous attempts, it still did not reach the accuracy levels we were aiming for. The classification score maxed out at **78%**, and the model wasn’t scalable for more granular classification.

---
![image](https://github.com/user-attachments/assets/7f2223a7-75c3-421d-8ecb-2c071a5652a9)

## **Switching to BERT: A Game-Changer** 🎯

After several unsuccessful attempts with traditional machine learning models, we realized that we needed something that could handle the complexity and contextual nature of Documents. That’s when we pivoted to **BERT**, a pre-trained transformer model that has revolutionized NLP tasks. Unlike traditional models, BERT understands context by processing the entire sentence or paragraph in one go rather than just individual words.

### **Why BERT?** 💬

- **Contextual Understanding** 🤓: BERT excels at understanding the relationships between words in a sentence, which is crucial for interpreting Documents where context is key (e.g., distinguishing between "Python Developer" and "Data Scientist").
- **Bidirectional Attention** 🔄: BERT reads the text in both directions (left-to-right and right-to-left), which makes it more effective at capturing context in long and complex sentences—common in Documents.
- **Pre-trained Knowledge** 🧠: BERT is pre-trained on vast datasets, meaning it already has an understanding of general language patterns, which we could fine-tune on our Document dataset for specific needs.

![Screenshot 2024-12-05 140303](https://github.com/user-attachments/assets/dcdc8434-3463-4cb2-91c2-f5a11b7ee4d3)

# Fine-Tuning BERT: A Dynamic, User-Driven Approach 🛠️

With BERT as the backbone, we have evolved our fine-tuning process to be more dynamic and user-driven. The model is now capable of adapting to **any number of classes** based on the dataset provided by the user, making it flexible for different classification tasks.

## **Stage 1: User-Uploaded Dataset** 📥  
In this iteration, the user can upload their own dataset, structured in a **ZIP file** containing text documents categorized into any number of classes. The dataset is processed using BERT’s pre-trained tokenizer to convert the text into a format suitable for the model. The model begins learning to classify documents based on the features present in the user’s dataset.

### Result:
- **Accuracy**: Depends on dataset quality and size 📊  
- **Precision**: User-defined 🔍  
- **Recall**: User-defined 🔁

## **Stage 2: Continuous Learning & Expansion** 📚  
As the user adds more data or new categories, the model continues to learn and refine its understanding. The system adapts to the new number of classes, ensuring that the model doesn’t need to be retrained from scratch. This continuous learning process enhances the model’s ability to classify increasingly diverse and complex documents.

### Result:
- **Accuracy**: Improves over time with more data 📊  
- **Precision**: Increases with more specific categories 🔍  
- **Recall**: Higher recall as model adapts 🔁

## **Stage 3: Dynamic Category Handling** 🔍  
The model can now handle **dynamic categorization** where the number of categories is not fixed. As the user adds new document types, the model learns to classify them appropriately without losing its ability to handle previous classes. This flexibility ensures that the model remains effective as the dataset evolves.

### Result:
- **Accuracy**: Continually improves 📊  
- **Precision**: Tailored to evolving categories 🔍  
- **Recall**: Optimized with incremental learning 🔁

## **Stage 4: Adaptive Specialization** 🎓  
As more specialized categories are introduced, the model can differentiate between nuanced document features. Whether it’s distinguishing between roles in the same domain or handling documents with intricate structures, BERT can adapt its understanding based on user input.

### Result:
- **Accuracy**: Improves with fine-tuned categories 📊  
- **Precision**: Reaches new heights 🔍  
- **Recall**: Focuses on niche distinctions 🔁

## **Stage 5: Ultimate Precision with Custom Categories** 🎯  
With the final stage, the model is capable of handling **highly specialized and unique categories** based on the user’s data. The model will also take into account **patterns and trends** such as career progression and skills evolution, providing insights tailored to the user’s needs.

### Result:
- **Accuracy**: Highly precise for custom data 📊  
- **Precision**: Excellent due to fine-tuning 🔍  
- **Recall**: Near-perfect as model adapts 🔁  
- **F1 Score**: Optimized for each dataset 💯

---
![image](https://github.com/user-attachments/assets/3414fcf7-474b-48cc-b748-7f3ebb1fafb5)

## **Key Advantages of the User-Driven Fine-Tuning Model**:
- **Dynamic Class Handling**: The model adapts to an unlimited number of classes, making it versatile for various domains.
- **Continuous Learning**: Allows for retraining with new data without starting from scratch.
- **Custom Adaptability**: Tailors to the user’s specific dataset and task requirements, improving accuracy and precision over time.
- **Real-Time Flexibility**: Users can upload new datasets and expand the classification capabilities at any time.


## **Performance Metrics: Accuracy Meets Innovation** 📊

Through each iteration, we saw incremental improvements in both classification accuracy and performance metrics. These weren’t just numbers—they were tangible results that reflected the system’s growing ability to understand and categorize Documents.

---
![image](https://github.com/user-attachments/assets/d38f791e-31ab-42c0-a8b0-b5cc7aad7747)


## **Why This Approach is Better** 🏆

The advantage of using **BERT** for fine-tuning over traditional models lies in its deep contextual understanding. The iterative process allowed us to:
- **Handle Complex Data** 🧩: Documents come in many formats and structures. BERT, fine-tuned over multiple iterations, was able to process these variations effectively.
- **Achieve High Accuracy** 🎯: Starting from a baseline accuracy of 80%, we achieved 92.5% accuracy through continuous fine-tuning. This marked a clear improvement over traditional models, which topped out at around 75%.
- **Scalability** 🌱: As we moved from broad categories to more granular classifications, the model demonstrated an ability to scale, making it suitable for diverse industries and job roles.

---


## **Key Features of the Model Training Process**:

- **Dynamic Learning** 🔄: At each stage, the model adapts to more granular data and refines its understanding of Documents.
- **Preprocessing & Tokenization** 📑: Using BERT’s tokenizer, we preprocessed thousands of Documents, converting them into a format that maintained the structure and meaning of the content.
- **Model Reusability** 🔁: After each iteration, we saved the state of the model, reloading and adapting it for the next phase, ensuring we retained all learning from previous stages.

---


## **Expanding the Model’s Potential** 🚀

Fine-tuning isn’t the end of the road—it’s just the beginning. The model can be expanded by:
- **Adding More Categories** ➕: Continuously expanding the number of job roles and classifications to capture an even wider variety of Documents.
- **Continuous Training** 🔄: As new Documents are processed, the model can be re-trained to stay current with industry trends and job market changes.
- **Incorporating Multi-Modal Data** 🖼️: Future iterations can integrate non-text data, such as job-related certifications and online portfolios, to provide a holistic view of each candidate.
- **IPFS Based Encryption Securty**
---


By leveraging **BERT’s advanced capabilities** and our detailed fine-tuning process, **ClassifyMe.ai** has evolved into a powerful, cutting-edge tool that continuously learns and adapts to provide the most accurate Document classification possible.


# 🌐 Web Platform: The User Interface That Brings AI to Life

Once the model was ready, it was time to bring it to life through a sleek, user-friendly web platform. We wanted **ClassifyMe.ai** to be more than just functional; we wanted it to be engaging, intuitive, and enjoyable to use.


---


## 👁️‍🗨️ Stunning User Interface (UI): A Platform that Pleases the Eye

ClassifyMe.ai isn't just powerful under the hood—it also offers an intuitive, visually appealing interface. With **React.js** and **Tailwind CSS**, the design is sleek, fast, and responsive, ensuring a smooth user experience across devices.

- **Seamless Upload**: Upload your Documents in PDF or DOCX format effortlessly. 📤
- **Instant Classification**: As soon as a Document is uploaded, it’s automatically classified into one of 96 categories. ⚡
- **Interactive Dashboard**: Users can explore the results with real-time visualizations, gaining deeper insights into the classification process. 📊

---

## 🎨 UI Components and Features:

- **Drag-and-Drop Interface**: Upload Documents with ease using a simple drag-and-drop area, making the entire process effortless. 🖱️
- **AI-Powered Analysis**: Instantly view detailed insights into the candidate’s skills, career trajectory, and recommended roles. 🤖
- **Real-time Confidence Scoring**: Track how confident the system is with each classification, fostering transparency in the AI decision-making process. 📈
- **Iteration-Based Visualization**: View how classifications evolve over time—each step is clearly marked, offering users transparency into how the model improves with each interaction. 🔄

---


## 🔧 Technical Architecture: Building the Backbone

### **Frontend**:
- **React.js** for a dynamic and responsive user experience. ⚛️
- **Redux** for seamless state management across the platform. 🔄
- **Tailwind CSS** ensures the platform looks as good as it functions, with modern, customizable designs. 🖌️

### **Backend**:
- **Django & Django REST Framework** for robust backend management and APIs. 🖥️
---



## 🛠️ How ClassifyMe.ai Works: Step-by-Step

1. **Upload Your Document**: Simply drag-and-drop your PDF or DOCX file onto the platform. 📤
2. **Instant Classification**: The AI-powered system immediately classifies the Document into one of 96 categories based on its content, such as "Software Engineer", "Data Scientist", or even more niche areas. 📋
3. **Visual Insights**: Watch as the platform generates a real-time classification confidence score, showing how sure the system is about its predictions. 📊
4. **Advanced Analysis**: Dive deeper into the AI-powered skill extraction and career trajectory mapping that helps both job seekers and recruiters gain valuable insights. 💡

---

## 🌈 Why ClassifyMe.ai is a Game-Changer?

- **Continuous Learning**: The system is designed to improve over time. As it processes more Documents, it fine-tunes its predictions, making the experience better for every user. 📚
- **Transparent AI**: You’re never left in the dark about how the AI is making decisions. Each classification step is visualized, letting you see the AI's reasoning in real-time. 🔍
- **Comprehensive Insights**: Beyond simply categorizing Documents, **ClassifyMe.ai** provides actionable insights like potential job role recommendations and skill assessments, helping you make more informed decisions. 📝

---

## 🚀 Join Us on This Journey!

ClassifyMe.ai isn't just an AI tool—it’s a transformation in how professionals engage with Documents. Whether you're a job seeker, recruiter, or developer, **ClassifyMe.ai** offers an unmatched level of intelligence, transparency, and ease of use. 🌟




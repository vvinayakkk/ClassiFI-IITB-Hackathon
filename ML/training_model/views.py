import os
import json
import shutil
import zipfile
import pandas as pd
import torch
import warnings
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
import datetime
import shutil
import os
import PyPDF2
import docx
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage


# Machine Learning Imports
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DocumentClassificationView:
    def __init__(self):
        # Initialize model and tokenizer paths
        self.model_path = os.path.join(settings.BASE_DIR, 'combined_category_model2')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Try to load existing model, if not exists, it will be None
        try:
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
        except:
            self.model = None

        # Load label mapping if exists
        self.label_mapping = self._load_label_mapping()

    def _load_label_mapping(self):
        """
        Load label mapping file
        """
        mapping_path = os.path.join(self.model_path, 'label_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                return json.load(f)
        return None

    def process_zip_file(self, zip_file):
        """
        Process uploaded zip file and create DataFrame with file details
        """
        # Create a temporary directory to extract files
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_upload')
        os.makedirs(temp_dir, exist_ok=True)

        # Extract zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Create DataFrame to store file information
        file_data = []
        unique_categories = set()

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    category = os.path.basename(root)  # Folder name as category
                    unique_categories.add(category)
                    
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_data.append({
                        'filename': file,
                        'category': category,
                        'text': content
                    })

        # Convert to DataFrame
        df = pd.DataFrame(file_data)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        # Create or update label mapping
        if self.label_mapping is None:
            self.label_mapping = {cat: idx for idx, cat in enumerate(sorted(unique_categories))}
            
            # Save label mapping
            mapping_path = os.path.join(self.model_path, 'label_mapping.json')
            with open(mapping_path, 'w') as f:
                json.dump(self.label_mapping, f)

        # Add numeric label
        df['label'] = df['category'].map(self.label_mapping)

        return df

    def predict_document_category(self, text):
        """
        Predict category for a single document
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Please train the model first.")

        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # Reverse lookup category name from ID
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return reverse_mapping[predicted_class_id]

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize the input text using BERT tokenizer
    """
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )

def compute_metrics(pred):
    """
    Compute evaluation metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    accuracy = accuracy_score(labels, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_document_classifier(training_df, base_model_path='./models'):
    """
    Train BERT model for document classification with timestamped folder
    """
    # Create timestamped folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(base_model_path, f'document_classifier_{timestamp}')
    
    # Ensure model directory exists
    os.makedirs(model_path, exist_ok=True)
    
    # Prepare train, validation split
    train, val = train_test_split(
        training_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=training_df['category']
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    num_labels = len(training_df['category'].unique())
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=num_labels
    )
    
    # Prepare label mapping
    label_mapping = {cat: idx for idx, cat in enumerate(sorted(training_df['category'].unique()))}
    
    # Convert dataframes to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    
    # Tokenize datasets
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer)
    
    train_dataset = train_dataset.map(tokenize_wrapper, batched=True)
    val_dataset = val_dataset.map(tokenize_wrapper, batched=True)
    
    # Set dataset format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=os.path.join(model_path, "logs"),
        logging_steps=10,
        save_total_limit=2,
        save_strategy="epoch",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save model, tokenizer, and label mapping
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save label mapping
    with open(os.path.join(model_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)
    
    print(f"Model training completed and saved to {model_path}")
    return model, tokenizer, label_mapping, model_path

@csrf_exempt
def train_classification_view(request):
    """
    View to handle training on uploaded zip file with timestamped model saving
    """
    if request.method == 'POST':
        try:
            # Get uploaded zip file
            zip_file = request.FILES.get('file')
            if not zip_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)

            # Save file temporarily
            file_path = default_storage.save('uploads/training_data.zip', zip_file)
            full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # Process zip file
            classifier = DocumentClassificationView()
            training_df = classifier.process_zip_file(full_file_path)

            # Train the model
            base_model_path = os.path.join(settings.BASE_DIR, 'models')
            os.makedirs(base_model_path, exist_ok=True)
            
            model, tokenizer, label_mapping, model_path = train_document_classifier(
                training_df, 
                base_model_path
            )

            return JsonResponse({
                'message': 'Model trained successfully',
                'total_files': len(training_df),
                'categories': list(training_df['category'].unique()),
                'model_path': model_path
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
@csrf_exempt
def predict_classification_view(request):
    """
    View to handle prediction on uploaded zip file of text files without predefined labels
    """
    if request.method == 'POST':
        try:
            # Get uploaded zip file
            zip_file = request.FILES.get('file')
            if not zip_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)

            # Save file temporarily
            file_path = default_storage.save('uploads/prediction_data.zip', zip_file)
            full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # Initialize classifier
            classifier = DocumentClassificationView()
            
            # Check if model is trained
            if classifier.model is None:
                return JsonResponse({
                    'error': 'No trained model found. Please train a model first.'
                }, status=400)

            # Create a temporary directory to extract files
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_prediction')
            os.makedirs(temp_dir, exist_ok=True)

            # Extract zip file
            with zipfile.ZipFile(full_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Prepare prediction results
            prediction_results = []

            # Process each text file
            for filename in os.listdir(temp_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(temp_dir, filename)
                    
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Predict category
                    predicted_category = classifier.predict_document_category(text_content)
                    
                    # Store prediction result
                    prediction_results.append({
                        'filename': filename,
                        'text': text_content,
                        'predicted_category': predicted_category
                    })

            # Clean up temporary directory
            shutil.rmtree(temp_dir)

            # Create results directory
            results_dir = os.path.join(settings.MEDIA_ROOT, 'prediction_results')
            os.makedirs(results_dir, exist_ok=True)

            # Create timestamped results file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f'prediction_results_{timestamp}.csv'
            results_path = os.path.join(results_dir, results_filename)

            # Save results to CSV
            results_df = pd.DataFrame(prediction_results)
            results_df.to_csv(results_path, index=False)

            return JsonResponse({
                'message': 'Prediction completed successfully',
                'total_files': len(prediction_results),
                'results_file': results_filename,
                'predictions': prediction_results
            })

        except ValueError as ve:
            # Specific error if model is not usable
            return JsonResponse({'error': str(ve)}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def predict_single_document_view(request):
    """
    View to handle prediction for a single uploaded document across all trained models
    """
    if request.method == 'POST':
        try:
            # Get uploaded text file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)

            # Ensure it's a text file
            if not uploaded_file.name.endswith('.txt'):
                return JsonResponse({'error': 'Only .txt files are supported'}, status=400)

            # Read file content
            file_content = uploaded_file.read().decode('utf-8')

            # Find all trained models
            models_dir = os.path.join(settings.BASE_DIR, 'models')
            model_predictions = []

            # Iterate through all model directories
            if os.path.exists(models_dir):
                for model_folder in os.listdir(models_dir):
                    full_model_path = os.path.join(models_dir, model_folder)
                    
                    # Check if it's a valid model directory
                    if os.path.isdir(full_model_path) and 'label_mapping.json' in os.listdir(full_model_path):
                        try:
                            # Load tokenizer and model
                            tokenizer = BertTokenizer.from_pretrained(full_model_path)
                            model = BertForSequenceClassification.from_pretrained(full_model_path)
                            model.eval()

                            # Load label mapping
                            with open(os.path.join(full_model_path, 'label_mapping.json'), 'r') as f:
                                label_mapping = json.load(f)

                            # Tokenize input
                            inputs = tokenizer(
                                file_content, 
                                return_tensors="pt", 
                                truncation=True, 
                                padding=True, 
                                max_length=512
                            )
                            
                            # Predict
                            with torch.no_grad():
                                outputs = model(**inputs)
                                logits = outputs.logits
                            
                            predicted_class_id = torch.argmax(logits, dim=1).item()
                            
                            # Reverse lookup category name from ID
                            reverse_mapping = {v: k for k, v in label_mapping.items()}
                            predicted_category = reverse_mapping[predicted_class_id]

                            model_predictions.append({
                                'model': model_folder,
                                'predicted_category': predicted_category
                            })

                        except Exception as model_error:
                            # Log the error but continue with other models
                            print(f"Error processing model {model_folder}: {str(model_error)}")
                            continue

            # If no models found
            if not model_predictions:
                return JsonResponse({
                    'error': 'No trained models found. Please train a model first.'
                }, status=400)

            # Prepare response
            return JsonResponse({
                'message': 'Document category predicted successfully',
                'filename': uploaded_file.name,
                'predictions': model_predictions
            })

        except ValueError as ve:
            # Specific error if model is not trained
            return JsonResponse({'error': str(ve)}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)



def extract_text_from_file(uploaded_file):
    """
    Extract text from various file types using built-in libraries
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    try:
        if file_extension == '.txt':
            return uploaded_file.read().decode('utf-8')
        
        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
        
        elif file_extension in ['.docx', '.doc']:
            doc = docx.Document(uploaded_file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        elif file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        
        elif file_extension == '.json':
            return uploaded_file.read().decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        raise ValueError(f"Error extracting text: {str(e)}")
@csrf_exempt
def comprehensive_document_analysis_view(request):
    """
    Comprehensive document analysis using Gemini with structured JSON response
    """
    if request.method == 'POST':
        try:
            # Get uploaded file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)

            # Extract text
            document_text = extract_text_from_file(uploaded_file)

            # Truncate text if too long (Gemini has token limits)
            MAX_TOKENS = 10000
            document_text = document_text[:MAX_TOKENS]

            # Initialize Gemini model
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                return JsonResponse({'error': 'Google API key not configured'}, status=500)

            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.3,
                max_tokens=2048,
                api_key=api_key
            )

            # Comprehensive Analysis Prompt
            comprehensive_prompt = PromptTemplate(
                input_variables=["document_text"],
                template="""Perform a comprehensive multi-dimensional analysis of the following document, 
                returning a STRICTLY STRUCTURED JSON response with the following keys:

1. document_overview: 
   - type: string (document classification)
   - purpose: string (primary purpose)
   - key_themes: list of strings

2. content_structure:
   - major_sections: list of strings
   - writing_style: string
   - tone: string
   - complexity: string (low/medium/high)

3. key_insights:
   - top_points: list of strings
   - critical_takeaways: list of strings
   - potential_implications: list of strings

4. linguistic_analysis:
   - vocabulary_complexity: string (basic/advanced/technical)
   - sentence_structure: string
   - readability_score: number (0-100)

5. contextual_classification:
   - suggested_domain: string
   - target_audience: string
   - professional_context: string

6. sentiment_analysis:
   - overall_sentiment: string (positive/negative/neutral)
   - emotional_undertones: list of strings
   - sentiment_ratio: number (-1 to 1)

7. potential_use_cases:
   - recommended_applications: list of strings
   - relevant_industries: list of strings
   - suggested_actions: list of strings

Analyze the document: {document_text}

Return ONLY a valid JSON response with these exact keys and corresponding structured data."""
            )

            # Create analysis chain
            analysis_chain = LLMChain(llm=llm, prompt=comprehensive_prompt)

            # Perform analysis
            analysis_result = analysis_chain.run(document_text)

            # Enhanced JSON parsing with multiple fallback strategies
            def parse_json_safely(json_str):
                # Remove any text before and after JSON
                import re
                json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                
                # Try parsing with multiple approaches
                try:
                    # First, standard parsing
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        # Try removing trailing/leading whitespace
                        return json.loads(json_str.strip())
                    except:
                        try:
                            # Use ast for more lenient parsing
                            import ast
                            return ast.literal_eval(json_str)
                        except:
                            # Final fallback to predefined structure
                            return {
                                "document_overview": {
                                    "type": "Unable to analyze",
                                    "purpose": "Analysis failed",
                                    "key_themes": ["N/A"]
                                },
                                "content_structure": {
                                    "major_sections": ["Unable to determine"],
                                    "writing_style": "Unknown",
                                    "tone": "Neutral",
                                    "complexity": "Medium"
                                },
                                "key_insights": {
                                    "top_points": ["Analysis could not be completed"],
                                    "critical_takeaways": ["Please review document manually"],
                                    "potential_implications": ["No implications determined"]
                                },
                                "linguistic_analysis": {
                                    "vocabulary_complexity": "Unknown",
                                    "sentence_structure": "Unable to assess",
                                    "readability_score": 0
                                },
                                "contextual_classification": {
                                    "suggested_domain": "Unclassified",
                                    "target_audience": "General",
                                    "professional_context": "Not specified"
                                },
                                "sentiment_analysis": {
                                    "overall_sentiment": "Neutral",
                                    "emotional_undertones": ["N/A"],
                                    "sentiment_ratio": 0
                                },
                                "potential_use_cases": {
                                    "recommended_applications": ["Manual review recommended"],
                                    "relevant_industries": ["Not determined"],
                                    "suggested_actions": ["Consult domain expert"]
                                }
                            }

            # Parse the analysis result
            parsed_analysis = parse_json_safely(analysis_result)
            print("analysis is:",parsed_analysis)
            # Prepare response
            return JsonResponse({
                'message': 'Document analysis completed successfully',
                'filename': uploaded_file.name,
                'analysis': parsed_analysis,
                'file_details': {
                    'original_name': uploaded_file.name,
                    'text_length': len(document_text)
                }
            })

        except ValueError as ve:
            return JsonResponse({'error': str(ve)}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
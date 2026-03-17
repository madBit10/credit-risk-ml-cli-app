# Credit Risk ML CLI App                                                                                                                                                                                   
                                                                                                                                                                                                               
  A machine learning CLI application that predicts whether a loan applicant will default or not.                                                                                                               
   
  ## Project Structure                                                                                                                                                                                         
                                                                                                                                                                                                             
  credit_risks_ml_cli_app/                                                                                                                                                                                     
  ├── data/                  # Dataset and exploration notebook                                                                                                                                              
  ├── src/                                                                                                                                                                                                     
  │   ├── main.py            # CLI entry point                                                                                                                                                                 
  │   ├── config/            # Configuration management                                                                                                                                                        
  │   ├── data/              # Data loading and preprocessing                                                                                                                                                  
  │   ├── models/            # Training and evaluation                                                                                                                                                       
  │   ├── pipeline/          # Training and prediction pipelines                                                                                                                                               
  │   └── utils/             # Logger and artifact saving                                                                                                                                                      
  ├── artifacts/                                                                                                                                                                                               
  │   └── metrics/           # Model performance metrics                                                                                                                                                       
  ├── tests/                 # Unit tests                                                                                                                                                                      
  ├── requirements.txt                                                                                                                                                                                         
  └── run_training.sh                                                                                                                                                                                          
                                                                                                                                                                                                               
  ## Dataset                                                                                                                                                                                                   
                                                                                                                                                                                                               
  - Source: Kaggle Credit Risk Dataset                                                                                                                                                                         
  - 32,581 rows, 12 columns                                                                                                                                                                                    
  - Target: `loan_status` (0 = no default, 1 = default)                                                                                                                                                        
                                                                                                                                                                                                               
  ## Model
                                                                                                                                                                                                               
  - Algorithm: Random Forest Classifier                                                                                                                                                                        
  - Handles class imbalance with `class_weight='balanced'`
  - Accuracy: 88.4%                                                                                                                                                                                            
  - Recall (defaulters): 77.2%                                                                                                                                                                                 
  - F1 Score (defaulters): 74.4%                                                                                                                                                                               
                                                                                                                                                                                                               
  ## Setup                                                                                                                                                                                                     
                                                                                                                                                                                                             
  ### 1. Clone the repository                                                                                                                                                                                  
  git clone
  cd credit_risks_ml_cli_app                                                                                                                                                                                   
                                                                                                                                                                                                             
  ### 2. Create virtual environment
  python3 -m venv venv
  source venv/bin/activate                                                                                                                                                                                     
   
  ### 3. Install dependencies                                                                                                                                                                                  
  pip install -r requirements.txt                                                                                                                                                                            
                                 
  ### 4. Add dataset
  Place `credit_risk_dataset.csv` in the `data/` folder.                                                                                                                                                       
                                                                                                                                                                                                               
  ## Usage                                                                                                                                                                                                     
                                                                                                                                                                                                               
  ### Train the model                                                                                                                                                                                        
  python3 -m src.main --train
                             
  ### Run tests then train
  bash run_training.sh
                                                                                                                                                                                                               
  ### Predict on new data
  python3 -m src.main --predict data/sample_input.csv                                                                                                                                                          
                                                                                                                                                                                                               
  ## Results
                                                                                                                                                                                                               
  | Metric | Value |                                                                                                                                                                                         
  |---|---|
  | Accuracy | 88.4% |
  | Recall (defaulters) | 77.2% |
  | Precision (defaulters) | 72.0% |                                                                                                                                                                           
  | F1 Score (defaulters) | 74.4% |
                                                                                                                                                                                                               
  ## Key Design Decisions                                                                                                                                                                                    
                                                                                                                                                                                                               
  - **Median imputation** for missing values — robust to outliers in financial data                                                                                                                            
  - **class_weight='balanced'** — prioritizes catching defaulters over overall accuracy
  - **Scaler saved as artifact** — ensures prediction uses same scaling as training                                                                                                                            
  - **Feature columns saved** — handles unseen categories in prediction data
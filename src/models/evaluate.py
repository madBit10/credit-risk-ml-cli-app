from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from src.utils.logger import get_logger
                                                                                         
logger = get_logger(__name__)
                                                                                         
def evaluate(model, X_test, y_test) -> dict:                                         
    """
    Evaluates the trained model on test data.
                                                                                        
    Args:
        model: Trained Random Forest model                                             
        X_test: Test features                                                        
        y_test: True test labels

    Returns:
        metrics: Dictionary of evaluation results
    """                                                                                
    logger.info("Evaluating model...")
                                                                                        
    y_pred = model.predict(X_test)                                                   

    accuracy = accuracy_score(y_test, y_pred)                                          
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()                                 
                                                                                        
    metrics = {
        "accuracy": accuracy,                                                          
        "classification_report": report,                                             
        "confusion_matrix": matrix
    }

    logger.info(f"Accuracy: {accuracy:.4f}")                                           
    logger.info(f"Recall (defaulters): {report['1']['recall']:.4f}")
    logger.info(f"F1 (defaulters): {report['1']['f1-score']:.4f}")                     
                                                                                        
    return metrics
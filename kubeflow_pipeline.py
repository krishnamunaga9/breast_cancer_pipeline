import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model_training import train_evaluate_model

@dsl.pipeline(
    name='Breast Cancer Classification Pipeline',
    description='An ML pipeline that performs data preprocessing, feature engineering, model training, and evaluation.'
)
def breast_cancer_pipeline():
    preprocess_op = create_component_from_func(
        func=preprocess_data,
        base_image='python:3.8',
        packages_to_install=['pandas', 'scikit-learn']
    )

    feature_engineer_op = create_component_from_func(
        func=feature_engineering,
        base_image='python:3.8',
        packages_to_install=['pandas', 'scikit-learn']
    )

    train_evaluate_op = create_component_from_func(
        func=train_evaluate_model,
        base_image='python:3.8',
        packages_to_install=['pandas', 'scikit-learn', 'joblib']
    )

    # Define pipeline steps
    preprocess_task = preprocess_op()
    feature_engineer_task = feature_engineer_op(X_train=preprocess_task.outputs['X_train'], X_test=preprocess_task.outputs['X_test'])
    train_evaluate_task = train_evaluate_op(X_train=feature_engineer_task.outputs['X_train_poly'], X_test=feature_engineer_task.outputs['X_test_poly'], y_train=preprocess_task.outputs['y_train'], y_test=preprocess_task.outputs['y_test'])

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(breast_cancer_pipeline, 'breast_cancer_pipeline.yaml')

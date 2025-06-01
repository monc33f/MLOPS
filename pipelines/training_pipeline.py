from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline
def training_pipeline(
    data_path1: str,
    data_path2: str
):
    # Step 1: Load the dataframes
    train_df, valid_df, test_df = ingest_df(data_path1=data_path1, data_path2=data_path2)

    # Step 2: Create generators
    tr_gen, valid_gen, ts_gen = clean_data(tr_df=train_df, valid_df=valid_df, ts_df=test_df)

    # Step 3: Train the model
    model = train_model(tr_df=tr_gen, valid_df=valid_gen)

    # Step 4: Evaluate the model
    _ = evaluate_model(model=model, tr_df=tr_gen, valid_df=valid_gen, ts_df=ts_gen)

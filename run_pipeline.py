from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Run pipeline with custom parameters
    training_pipeline(
        data_path1="data1/training",
        data_path2="data1/testing",
    )
    
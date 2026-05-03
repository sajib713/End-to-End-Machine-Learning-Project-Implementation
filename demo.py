from youth_migration.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        print(" Training Pipeline Started...")

        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        print("Training Pipeline Finished!")

    except Exception as e:
        print("Error:", e)
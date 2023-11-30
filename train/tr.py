import openai
# easy sync 

openai.api_key = "your api key" 
openai.api_base = "your endpoint URL"

openai.api_type = 'azure'
openai.api_version = '2023-09-15-preview' # This API version or later is required to access fine-tuning for turbo/babbage-002/davinci-002


from openai.wandb_logger import WandbLogger

WandbLogger.sync(project="OpenAI-Fine-Tune")

# Assuming the training_file_id and validation_file_id are already set
training_file_id = "your uploaded train file id"
validation_file_id = "your uploaded val file id"

# Create a new fine-tuning job
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-35-turbo-0613" 
)

job_id = response["id"]

print("Job ID:", job_id)
print("Status:", response["status"])
print(response)


import os
import torch
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import pandas as pd
import json
from typing import Dict, List

class MedicalChatbotTrainer:
    def __init__(self, model_name: str = "unsloth/tinyllama-1.1b-bnb-4bit"):
        self.model_name = model_name
        self.max_seq_length = 2048
        self.dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        
        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=True,
        )
        
    def prepare_model_for_training(self):
        """Prepare model for LoRA fine-tuning"""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # Choose any number > 0 for LoRA
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
    def create_medical_prompt(self, instruction: str, input_text: str = "", response: str = ""):
        """Create a standardized prompt format for medical conversations"""
        if input_text:
            prompt = f"""Below is a medical instruction that describes a task, paired with input. Write an appropriate response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}"""
        else:
            prompt = f"""Below is a medical instruction. Write an appropriate response.

### Instruction:
{instruction}

### Response:
{response}"""
        return prompt
    
    def prepare_dataset(self, data_path: str = None):
        """Prepare medical dataset for training"""
        
        # Sample medical Q&A data - replace with your actual dataset
        if data_path and os.path.exists(data_path):
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                data = df.to_dict('records')
        else:
            # Sample medical data for demonstration
            data = [
                {
                    "instruction": "What are the symptoms of diabetes?",
                    "input": "",
                    "output": "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurred vision, slow-healing wounds, and frequent infections. Type 1 diabetes symptoms often develop quickly, while Type 2 symptoms may develop gradually."
                },
                {
                    "instruction": "Explain hypertension and its risks",
                    "input": "",
                    "output": "Hypertension, or high blood pressure, occurs when blood pressure consistently measures 140/90 mmHg or higher. It's often called the 'silent killer' because it typically has no symptoms. Risks include heart disease, stroke, kidney damage, and vision problems. Management includes lifestyle changes and medication when necessary."
                },
                {
                    "instruction": "What should I do for a fever?",
                    "input": "Patient has a temperature of 102°F",
                    "output": "For a fever of 102°F: 1) Stay hydrated with plenty of fluids, 2) Rest and avoid strenuous activity, 3) Use fever reducers like acetaminophen or ibuprofen as directed, 4) Dress lightly and use cool compresses. Seek medical attention if fever exceeds 103°F, persists more than 3 days, or is accompanied by severe symptoms."
                }
            ]
        
        # Format data into training prompts
        formatted_data = []
        for item in data:
            prompt = self.create_medical_prompt(
                instruction=item.get("instruction", ""),
                input_text=item.get("input", ""),
                response=item.get("output", "")
            )
            formatted_data.append({"text": prompt})
        
        return Dataset.from_list(formatted_data)
    
    def train(self, dataset_path: str = None, output_dir: str = "./medical_chatbot_model"):
        """Train the medical chatbot"""
        
        # Prepare model
        self.prepare_model_for_training()
        
        # Prepare dataset
        dataset = self.prepare_dataset(dataset_path)
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,  # Increase for better results
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=50,
            save_total_limit=2,
        )
        
        # Create trainer
        from trl import SFTTrainer
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return output_dir

def main():
    """Main training function"""
    trainer = MedicalChatbotTrainer()
    
    # Train the model
    model_path = trainer.train(
        dataset_path="medical_dataset.json",  # Path to your medical dataset
        output_dir="./medical_chatbot_model"
    )
    
    print(f"Training completed! Model saved to: {model_path}")
    
    # Test the trained model
    print("\nTesting the model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Test query
    test_prompt = trainer.create_medical_prompt(
        instruction="What are the early signs of heart disease?",
        input_text="",
        response=""
    )
    
    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Test Response:")
    print(response.split("### Response:")[-1].strip())

if __name__ == "__main__":
    main()
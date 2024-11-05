import os
import json
import pdfplumber
import openai

# Set your OpenAI API key
# openai.api_key = 'open ai key'  # Replace with your OpenAI API key

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to split text into chunks
def split_text(text, max_length=2000):  # Adjust max_length based on needs
    sentences = text.split('\n')  # Split by new line; you can change this to a different delimiter if needed
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to generate QA pairs from text
def generate_qa(text):
    prompt = f"Generate as many question-answer pairs as possible based on the following text:\n\n{text}\n\n"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Ensure you're using the correct model
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500  # Adjust max_tokens based on expected output length
        )
        
        qa_pairs = response.choices[0].message['content'].strip()
        return qa_pairs.split('\n')  # Assuming each QA pair is on a new line

    except Exception as e:
        print(f"Error generating QA for passage: {str(e)}")
        return []

# Main function to process multiple PDFs and generate the QA dataset
def generate_qa_dataset(pdf_directory, output_file):
    qa_dataset = []

    # Loop through each PDF file in the specified directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Processing: {pdf_path}")

            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_path)

            # Split the text into manageable chunks
            text_chunks = split_text(text)

            # Generate QA pairs for each chunk
            for chunk in text_chunks:
                if chunk:  # Only generate QA if the chunk is not empty
                    qa_pairs = generate_qa(chunk)
                    if qa_pairs:
                        qa_dataset.append({
                            "file": filename,
                            "qa_pairs": qa_pairs
                        })

    # Save the dataset to a JSON file
    with open(output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=2)
    print(f"Dataset saved to {output_file}")

# Set your directory containing PDFs and output file name
pdf_directory = '/home/hamza/Desktop/gptfinetune/books'  # Your PDF directory
output_file = 'qa_dataset.json'  # Output file name

# Generate the QA dataset
generate_qa_dataset(pdf_directory, output_file)

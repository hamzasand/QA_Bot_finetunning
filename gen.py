import os
import csv
import pdfplumber
import openai

# Set your OpenAI API key
# openai.api_key = 'open ai key'  # Replace with your OpenAI API key

# Function to extract text from a single PDF page using pdfplumber
def extract_text_from_pdf_page(pdf, page_num):
    page = pdf.pages[page_num]
    return page.extract_text() if page else ""

# Function to split text into chunks
def split_text(text, max_length=1500):  # Reduce max_length for lower memory usage
    sentences = text.split('\n')
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

# Function to generate concise and accurate QA pairs from text
def generate_qa(text):
    prompt = (
        f"Generate concise and accurate question-answer pairs based on the following text. "
        f"Format each question-answer pair as follows:\n\n"
        f"Q: [question]\nA: [answer]\n\n"
        f"Only provide factual answers, and keep answers concise (2-3 sentences at most).\n\n"
        f"Text:\n{text}\n\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        qa_pairs = response.choices[0].message['content'].strip()
        qa_list = []

        # Split response by line and parse into question-answer pairs
        lines = qa_pairs.splitlines()
        question, answer = None, None

        for line in lines:
            if line.startswith("Q:"):
                if question and answer:
                    qa_list.append((question.strip(), answer.strip()))  # Append previous QA pair
                question = line[2:].strip()  # New question
                answer = None  # Reset answer

            elif line.startswith("A:") and question:
                answer = line[2:].strip()  # New answer

        # Append the last QA pair if exists
        if question and answer:
            qa_list.append((question.strip(), answer.strip()))

        return qa_list

    except Exception as e:
        print(f"Error generating QA for passage: {str(e)}")
        return []

# Main function to process multiple PDFs and generate the QA dataset in CSV format
def generate_qa_dataset(pdf_directory, output_file):
    # Open CSV file for writing
    with open(output_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Question', 'Answer'])  # Write header

        # Loop through each PDF file in the specified directory
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                print(f"Processing: {pdf_path}")

                with pdfplumber.open(pdf_path) as pdf:
                    # Process each page to limit memory use
                    for page_num in range(len(pdf.pages)):
                        text = extract_text_from_pdf_page(pdf, page_num)
                        if not text:
                            continue

                        # Split the text into smaller chunks
                        text_chunks = split_text(text)

                        # Generate QA pairs for each chunk
                        for chunk in text_chunks:
                            if chunk:  # Only generate QA if the chunk is not empty
                                qa_pairs = generate_qa(chunk)
                                for question, answer in qa_pairs:
                                    csv_writer.writerow([question, answer])  # Write to CSV

    print(f"Dataset saved to {output_file}")

# Set your directory containing PDFs and output file name
pdf_directory = '/home/hamza/Desktop/gptfinetune/book1'  # Your PDF directory
output_file = 'qa_dataset.csv'  # Output file name

# Generate the QA dataset
generate_qa_dataset(pdf_directory, output_file)

import PyPDF2
import PyPDF2.errors


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
    pdf_path (str): Path to the PDF file.
    
    Returns:
    str: Extracted text from the PDF.
    """
    text = ""
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Iterate through all pages
            for page in pdf_reader.pages:
                # Extract text from the page and add it to our text string
                text += page.extract_text() + "\n"
        
        return text.strip()
    
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
    except PyPDF2.errors.PdfReadError:
        print(f"Error: {pdf_path} is not a valid PDF file or it's encrypted.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
    return ""  # Return empty string if extraction fails
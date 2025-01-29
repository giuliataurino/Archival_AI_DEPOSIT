import Levenshtein
import xml.etree.ElementTree as ET
from pathlib import Path
import csv
import pandas as pd


def compute_levenstein_distance(ai_transcription,ground_truth_transcription):
    return Levenshtein.distance(ai_transcription,ground_truth_transcription)

def extract_unicode_transcription(xml_file):
    """
    Extracts and returns the concatenated text from all <Unicode> elements inside <TextEquiv> from a PAGE XML file.
    
    :param xml_file: Path to the XML file
    :return: name of the file, A single string containing all extracted text
    """
    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Define the namespace from the XML
    namespace = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    
    page_elem = root.find(".//ns:Page", namespaces=namespace)
    filename = page_elem.get('imageFilename')
    # Extract all <Unicode> elements within <TextEquiv>
    transcriptions = [unicode_elem.text for unicode_elem in root.findall(".//ns:Unicode", namespaces=namespace) if unicode_elem.text]
    
    # Join into a single transcription if needed
    return filename," ".join(transcriptions)

def generate_transkribus_transcriptions_csv(path_to_xml_files):
    """
    Helper Method to aggregate the transcriptions from a Transkribus generated xml_file directory
    """
    page_dir = Path(path_to_xml_files)
    xml_files = list(page_dir.glob("*.xml"))
    with open('transkribus_transcriptions_csv','w', newline = '',encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for xml_file in xml_files:
            xml_file_name, unicode_transcription = extract_unicode_transcription(xml_file)
            writer.writerow([xml_file.name,unicode_transcription])

def process_text(text):
    """
    Processes text to be compatible for an accurate levenstein distance computation
    :param text: text to be processed
    :return: processed version of the text
    """
    """
    Processes text to be compatible for an accurate Levenshtein distance computation
    :param text: text to be processed
    :return: processed version of the text
    """
    if text is None:
        return ""
        
    # Convert to uppercase for consistent comparison
    text = text.upper()
    
    # Standardize date formats (JULY -> JUL)
    month_mappings = {
        "JULY": "JUL",
        "JUNE": "JUN",
        "JANUARY": "JAN",
        "FEBRUARY": "FEB",
        "MARCH": "MAR",
        "APRIL": "APR",
        "AUGUST": "AUG",
        "SEPTEMBER": "SEP",
        "OCTOBER": "OCT",
        "NOVEMBER": "NOV",
        "DECEMBER": "DEC"
    }
    for full, abbrev in month_mappings.items():
        text = text.replace(full, abbrev)
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Remove punctuation except hyphen
    text = ''.join(char for char in text if char.isalnum() or char.isspace() or char == '-')
    
    # Standardize common variations
    replacements = {
        "SKYLINE": "SKY LINE",
        "FULLMOON": "FULL MOON",
        "BJ": "BU"
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
        
    return text


def compare_ai_to_groundtruth(ai_output_csv, transkribus_transcriptions_csv):
    """
    Compares AI generated transcriptions to ground truth transcriptions from Transkribus
    and computes Levenshtein distance scores
    
    :param ai_output_csv: CSV file containing AI generated transcriptions
    :param transkribus_transcriptions_csv: CSV file containing ground truth transcriptions 
    """
    # Create dictionaries to store transcriptions
    transkribus_dict = {}
    ai_dict = {}
    
    # Read ground truth transcriptions
    with open(transkribus_transcriptions_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            file_name = row[0][5:-4] #Remove .xml at the end of the file_name and 0101_ from the beginning
            transcription = row[1]
            transkribus_dict[file_name] = transcription
    print(transkribus_dict)

    # Read AI transcriptions            
    with open(ai_output_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            file_name = row[0] 
            ai_transcription = row[4]
            ai_dict[file_name] = ai_transcription
    
    
    
    # Compare transcriptions and write results
    with open('transcription_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Ground Truth Transcription', 'AI Transcription', 'Levenshtein Distance'])
        for filename in ai_dict:
            if filename in transkribus_dict:
                distance = compute_levenstein_distance(ai_dict[filename], transkribus_dict[filename])
                writer.writerow([filename, transkribus_dict[filename], ai_dict[filename], distance])
        """
        for filename in transkribus_dict:
            if filename in ai_dict:
                distance = compute_levenstein_distance(ai_dict[filename], transkribus_dict[filename])
                writer.writerow([filename, transkribus_dict[filename], ai_dict[filename], distance])
        """
        
if __name__ == "__main__":
    compare_ai_to_groundtruth('Qwen_first_50_Outputs.csv', 'transkribus_transcriptions_csv')
    


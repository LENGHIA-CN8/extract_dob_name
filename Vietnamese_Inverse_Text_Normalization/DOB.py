import sys
sys.path.append('../')
import logging
from Vietnamese_Inverse_Text_Normalization.inverse_normalize import inverse_normalize
import re
from datetime import datetime
# Regular expression pattern to match dates in the format "dd/mm yy" and "dd/mm yyyy"
replace_date_pattern = r"(\d{1,2}/\d{1,2}) ?\/?(\d{2,4})"
date_vocab = ['không','ngày', 'tháng', 'năm', 'một', 'mốt', 'hai', 'ba', 'bốn', 'tư', 'năm', 'lăm', 'nhăm', 'sáu', 'bảy', 'bẩy', 'tám', 'chín', 'mười', 'trăm', 'mươi', 'nghìn', 'chục', 'linh', 'ngàn', 'lẻ']
current_year = datetime.now().year
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(message)s')
logger = logging.getLogger(__name__)

# check vocab 
def clean_text(text):
    text_af = ''
    for w in text.split():
        if w in date_vocab:
            text_af = text_af + w + ' '
    return text_af

# Function to replace the matched dates with the desired format
def replace_date(match):
    day_month = match.group(1)
    year = match.group(2)

    # If the year has two digits, prepend '19' to it
    if len(year) == 2:
        year = f"19{year}"

    return f"{day_month}/{year}"

def check_format_date(text):
    pattern = r'^ngày (\d{1,2}/\d{1,2}/\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2} ?\d{2}|\d{1,2}/\d{1,2} ?\d{4})$'
    
    if re.match(pattern, text):
        return text
    else:
        return None

# Check type tháng mười một một chín chín ba is duplicate or not
def check_year(text):
    logger.info(f'Inside year: {text}')
    if text.split(' ')[1] == 'năm':
        return text, None
    c_text = inverse_normalize(text)
    four_digit = r'\b\d{4}\b'
    if re.match(four_digit, c_text) and int(c_text) > 1200:
        return text, None
    
    five_digit = r'\b\d{5}\b'
    if re.match(five_digit, c_text):
        text = text.split()
        text.insert(1, 'năm')
        text = ' '.join(text)
        return text, 5  
    
    return text, None
    
def check_format_string(text):
    if 'tháng' not in text or text.split()[0] == 'tháng':
        return None
    
    ## hai năm tháng ba năm .. => hai lăm tháng ba năm ..
    pattern = r"\bnăm\b(?= tháng)"
    text = re.sub(pattern, "lăm", text)
    
    ## check type tháng mười một một
    # pattern = r'(?<=tháng )\bmười\b'
    pattern = r'\btháng mười\b(?=.*\b\w+\b)|\btháng một\b(?=.*\b\w+\b)'
    match = re.search(pattern, text)
    mod_text = None
    if match:
        mod_text, is_5digit = check_year(text[match.span()[1]:].strip())   
        if mod_text and is_5digit:
            text = text[:match.span()[1]+1] + mod_text
            pattern = r'\btháng một\b'
            text = re.sub(pattern, 'tháng mười', text)
        else:
            text = text[:match.span()[1]+1] + mod_text
        
    if 'ngày' not in text:
        text = 'ngày ' + text.strip() 
    return text

def extract_date(text):
    # Regular expression pattern to match dates in the formats "dd/mm/yyyy", "dd/mm", and "dd"
    date_pattern = r"(\d{1,2})(?:/(\d{1,2})(?:/(\d{4}))?)?"

    # Search for the date pattern in the text
    match = re.search(date_pattern, text)

    if match:
        day, month, year = match.groups()
        if int(day) > 31:
            return None
        if int(day) > 30 and int(month) == 2:
            return None
        if int(month) > 12:
            return None
        if int(year) > current_year:
            return None
        # If the year is present and the month is present, return the date in the "dd/mm/yyyy" format
        if year and month:
            return f"{day}/{month}/{year}"
        
        # Return None if only the day or the month is present
        return None

    # Return None if there's no match
    return None

def extract_dob(text):
    text = clean_text(text)
    text = check_format_string(text)
    logger.info(f'Before inverse: {text}')
    if text:
        text = inverse_normalize(text)
    else:
        return None
    
    logger.info(f"Text after inverse_normalize: {text}")
    text = re.sub('\.', '', text)
    
    # check format date is dd/mm/yy or dd/mm/yyyy
    text = check_format_date(text)
    logger.info(f'After check format date: {text}')
    if text is None:
        return None
    
    text = re.sub(replace_date_pattern, replace_date, text)
    text = extract_date(text)
    
    return text
    
if __name__=="__main__":
    text = 'ngày hai mươi ba tháng một một chín chín tư năm'
    print(inverse_normalize(''))
    print(extract_dob(text))
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import time
from OCR import query

no_of_chars = 70

def main():
    st.set_page_config(
        page_title="AI Handwriter Lekhak",
        page_icon="✨",
    )

    pass

if __name__ == "__main__":
    
    main()

# UI
st.title("AI Handwriter Lekhak 🌟")
st.subheader("AI for writing in your handwriting")
st.markdown(
    """
    <style>
        .reportview-container {
            width: 90%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

standard_height = 16
standard_width = 16


def store_char(selected_contour_with_position,binary_image,char_images):
  _,x,y,w,h,contour_index=selected_contour_with_position
  print(x,y,w,h,contour_index)

  
  character_shape = binary_image[y:y+h, x:x+w]
  resized_character = cv2.resize(character_shape, (standard_width, standard_height))
  character_pil_image = Image.fromarray(character_shape)

  _, temp_file_name = tempfile.mkstemp(suffix=".jpg")

  
  # Save the image to a JPG file
  character_pil_image.save(temp_file_name)

  st.image(temp_file_name)

  prediction_response = query(temp_file_name)

  print(prediction_response)
  # input()
  result=' '

  if 'error' in prediction_response:
      return

  if len(prediction_response) > 0 :
    result = prediction_response[0]['generated_text']
    print(result,result[0])

  
  current_char = st.text_input(label="Identified Character : ", value =result[0],key=str(time.time()))
  
  print("current_char : ",current_char)

  if len(current_char)>0 and current_char[0].isalnum():
    char_images[current_char[0]]=resized_character
    print("updated char_images: ",char_images.keys())

def load_sample_writing(uploaded_file):
    if "char_images" not in st.session_state:
        # Initialize an empty dictionary if not present
        st.session_state.char_images = {}
    print("retrived from session_state : ",st.session_state.char_images.keys())
    char_images=st.session_state.char_images

    # st.title("Image Uploader")
    
    # # File uploader widget to upload image
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        #  Read the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Thresholding image
        _, thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

        # inverting image to binary with white background and black foreground
        binary_image = cv2.bitwise_not(thresh)

        # Displaying binary image
        st.image(binary_image, caption='Loaded Sample Writing', use_column_width=True)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print("len of contours : ",len(contours))

        # Sort contours from left to right, top to bottom
        contours = sorted(contours, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))

        contours_with_positions=[]
        
        contour_index=-1

        for contour in contours:
            contour_index += 1
            x, y, w, h = cv2.boundingRect(contour)
            contours_with_positions.append([w*h,x,y,w,h,contour_index])



        # Sort the list by decending order of area of contours (product of x and y)
        
        selected_contours_with_positions = sorted(contours_with_positions, key=lambda x: x[0],reverse=True)
        
        for selected_contour_with_position in selected_contours_with_positions[0:no_of_chars]:
            _,x,y,w,h,contour_index=selected_contour_with_position
            print(x,y,w,h,contour_index)

            
            character_shape = binary_image[y:y+h, x:x+w]
            resized_character = cv2.resize(character_shape, (standard_width, standard_height))
            character_pil_image = Image.fromarray(character_shape)

            _, temp_file_name = tempfile.mkstemp(suffix=".jpg")

            
            # Save the image to a JPG file
            character_pil_image.save(temp_file_name)

            st.image(temp_file_name)

            prediction_response = query(temp_file_name)

            print(prediction_response)
            
            if 'error' in prediction_response:
                continue
            # input()
            result=' '

            if len(prediction_response) > 0 :
                result = prediction_response[0]['generated_text']
                print(result,result[0])

            
            current_char = st.text_input(label="Identified Character : ", value =result[0],key=str(time.time()))
            
            print("current_char : ",current_char)

            if len(current_char)>0 and current_char[0].isalnum():
                print("in ",contour_index)
                print("current_char[0] not in char_images.keys() : ",(current_char[0] not in char_images.keys()))
                if current_char[0] not in char_images.keys():
                    char_images[current_char[0]]=resized_character
                    print("updated char_images: ",char_images.keys())
            # store_char(selected_contour_with_position,binary_image,char_images)
            # input("Press key to continue")
            pass
        
        st.session_state.char_images=char_images
        print("added into session_state : ",st.session_state.char_images.keys())

    pass

st.markdown("##### Upload your sample handwriting")

# File uploader widget to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

char_images={}
if "char_images" in st.session_state:
    char_images=st.session_state.char_images

if st.button('Load your handwriting') and len(char_images.keys()) <= 0 and uploaded_file is not None:
    load_sample_writing(uploaded_file)

paragraph=st.text_area('Enter your text here')

if st.button('Generate handwritten'):
    print("char_image : ",char_images.keys())

    word_count = len([word for word in paragraph.split() if word.strip()])
    no_of_lines=word_count//5 + word_count//10 + 10
    print(word_count,no_of_lines)
    import cv2
    import numpy as np

    # Assuming you have a dictionary char_images containing alphanumeric characters as keys and their corresponding images as values
    # Example: char_images = {'A': image_of_A, 'B': image_of_B, ...}

    # Define standard height and width for characters
    standard_height = 16
    standard_width = 16

    # Define the maximum width of the output canvas
    max_width = 50 * standard_width
    max_height = no_of_lines * standard_height
    # Create a white canvas to output the paragraph
    output_canvas = np.ones((max_height, max_width), dtype=np.uint8) * 255  # White background

    # Initialize coordinates for drawing on the output canvas
    x_coord = 0
    y_coord = 0

    # Define space to leave between characters
    space_width = standard_width//3

    # Given paragraph
    # paragraph = """Shri Ram, an embodiment of righteousness and compassion, is a central figure in Hindu mythology and revered as the seventh avatar of Lord Vishnu. Born as the prince of Ayodhya to King Dasharatha and Queen Kaushalya, his life exemplifies the pursuit of dharma (righteousness) and the triumph of good over evil.

    # The narrative of Ramayana, an ancient epic, chronicles his journey from princely opulence to exile in the forest, enduring numerous trials and tribulations. Alongside his devoted wife Sita and loyal brother Lakshmana, Ram navigates the complexities of duty and devotion, facing adversities with unwavering resolve.

    # """
    

    # Iterate over each character in the paragraph
    for char in paragraph:

        if x_coord + standard_width > max_width:
            x_coord = 0
            y_coord += standard_height + space_width
            
        if char == ' ':
            # Leave space for a standard-sized character horizontally
            x_coord += standard_width + space_width
            
            # Check if wrapping to the next line is needed
            if x_coord + standard_width > max_width:
                x_coord = 0
                y_coord += standard_height + space_width

        elif char == '\n':
            # Move to the next line
            x_coord = 0
            y_coord += standard_height + space_width

        elif char.isalnum():
            # Get the image of the alphanumeric character
            char_image = char_images.get(char)  # Assuming char_images keys are uppercase alphanumeric characters
            
            if char_image is None:
                char_image = char_images.get(char.upper())
            if char_image is None:
                char_image = char_images.get(char.lower())

            if char_image is not None:
                # Paste the character image onto the output canvas
                resized_character = cv2.resize(char_image, (standard_width, standard_height))
                output_canvas[y_coord:y_coord+standard_height, x_coord:x_coord+standard_width] = resized_character
                
                # Move horizontally to the next position
                x_coord += standard_width + space_width
                
                # Check if wrapping to the next line is needed
                if x_coord + standard_width > max_width:
                    x_coord = 0
                    y_coord += standard_height + space_width
            
            else:
                # Leave space for a standard-sized character horizontally
                x_coord += standard_width + space_width
                
                # Check if wrapping to the next line is needed
                if x_coord + standard_width > max_width:
                    x_coord = 0
                    y_coord += standard_height + space_width
        
        else:
            # Handle other characters (e.g., punctuation) if needed
            pass

    st.image(output_canvas,caption='Generated Image')    
    pass
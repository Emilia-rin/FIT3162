import io
import os
import sys
import glob
import csv
import cv2
import joblib
import gradio as gr
import pandas as pd
import numpy as np
from os.path import join
from os.path import splitext
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from moviepy.editor import VideoFileClip

CURRENT_MODEL = 'multimodal_mexp_and_gaze_02.pkl'

######################################### Helper Functions ##########################################################

# mp4 convertor
def convert_mov_to_mp4(mov_file_path):
    # Extract directory and filename without extension
    directory = os.path.dirname(mov_file_path)
    file_name = os.path.splitext(os.path.basename(mov_file_path))[0]
    output_mp4_file_path = os.path.join(directory, file_name + '.mp4')
    
    # Convert the video file
    video_clip = VideoFileClip(mov_file_path)
    video_clip.write_videofile(output_mp4_file_path, codec='libx264')
    video_clip.close()
    
    print("the new path of mp4 file is "+str(video_clip))
    return output_mp4_file_path

######################################### Datasets Functions ##########################################################

# 1. compute total training datasets distribution based on gender bias to show more datasets information
def count_gender_distribution():
    csv_path = r"dataset_file\\gender_bias_trial_data_results.csv" 
    gender_bias_file = pd.read_csv(csv_path)
    
    # Extract gender and video count data
    gender = gender_bias_file['Gender']
    video_count = gender_bias_file['Video_Count']
    
    # Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(video_count, labels=gender, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
    
    return fig

# 2. compute total training datasets distribution based on deceptive and truthful labels
def count_training_distribution():
    
    csv_list = {
        r"dataset_file\\Annotation_mexp_features.csv",
        r"dataset_file\\Annotation_gaze_features.csv"    
    }
    all_labels = []
    
    for file in csv_list:
        current_file = pd.read_csv(file)
        all_labels.append(current_file['label'])
    
    combined_labels = pd.concat(all_labels, ignore_index=True)
    label_counts = combined_labels.value_counts()
    
    # Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
    ax.axis('equal')
    ax.set_title('Label Distribution')
    return fig
        
# 3. compute the total training datasets amount (add up for every new datasets added)
def count_dataset()->int:
    total = 0
    total += 121
    total += 352
    return total

# 4. compute model accuracy
def compute_accuracy():
    return 0.6526315789473685 * 100

######################################### Deception Functions #########################################################

# extract gaze features into csv file
def extract_gaze_features(video_path):
    #openface_cmd ="/Users/jingweiong/openFace/OpenFace/build/bin/"         # onji
    openface_cmd ="D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"        # jk

    # Extract the directory and filename from the video path
    directory, filename = os.path.split(video_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Define the output path for the CSV
    output_csv = os.path.join(directory, f"Gaze_{base_filename}.csv")

    # Construct the command to run feature extraction with gaze tracking
    cmd = f"{openface_cmd} -f \"{video_path}\" -out_dir \"{os.path.dirname(output_csv)}\" -of \"{output_csv}\" -gaze"

    # Execute the command
    result = os.system(cmd)
    
    if result == 0: print(f"Gaze data extracted successfully for {filename}, saved to {output_csv}")
    else: print(f"Failed to extract gaze data for {filename}")
    return output_csv

# extract micro expression features into csv file
def extract_micro_features(video_path):
    #openface_cmd ="/Users/jingweiong/openFace/OpenFace/build/bin/"                             # onji
    openface_cmd ="D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"      # jk

    directory, filename = os.path.split(video_path)
    base_filename = os.path.splitext(filename)[0]    
    output_csv = os.path.join(directory, f"Mexp_reallifedeception_{base_filename}.csv")
    cmd = f"{openface_cmd} -f \"{video_path}\" -out_dir \"{os.path.dirname(output_csv)}\" -of \"{output_csv}\" -pose -aus"
    result = os.system(cmd)
    
    if result == 0: print(f"Mexp data extracted successfully for {filename}, saved to {output_csv}")
    else: print(f"Failed to extract mexp data for {filename}")
    return output_csv

# get loaded model (change the CONSTANT var based on latest model trained)
def get_model():
    model = joblib.load(CURRENT_MODEL)
    return model

# pre processing with PCA
def preprocess_data_with_pca(filepath, n_samples, expected_features):
    data = pd.read_csv(filepath)
    data = data.drop(columns=["Unnamed: 0", "frame", "label", "face_id", "timestamp", "confidence", "success"], errors='ignore')
    data = data.drop_duplicates()

    # Resample to a fixed number of samples
    if len(data) > n_samples:
        data = resample(data, n_samples)
    elif len(data) < n_samples:
        repeat_factor = n_samples // len(data) + 1
        data = pd.DataFrame(np.tile(data, (repeat_factor, 1)), columns=data.columns)[:n_samples]

    # PCA for dimensionality reduction if the number of features is more than expected
    if data.shape[1] > expected_features:
        pca = PCA(n_components=expected_features)
        data = pca.fit_transform(data)
    elif data.shape[1] < expected_features:
        raise ValueError(f"Data has fewer features ({data.shape[1]}) than expected ({expected_features}).")

    return data

# prediction input function
def predict_inp(video_path, svm_model, gaze_features=292, mexp_features=45):
    
    # gaze_filepath = r"D:\\fit3162\\dataset\\output_gaze\\Gaze_reallifedeception_trial_lie_042.csv"
    # mexp_filepath = r"D:\\fit3162\\dataset\\output_micro_expression\\Mexp_reallifedeception_trial_lie_042.csv"
    gaze_filepath = extract_gaze_features(video_path)
    mexp_filepath = extract_micro_features(video_path)
                
    # Preprocess gaze data / microexpression data with PCA
    gaze_data = preprocess_data_with_pca(gaze_filepath, n_samples=300, expected_features=gaze_features)
    mexp_data = preprocess_data_with_pca(mexp_filepath, n_samples=300, expected_features=mexp_features)

    # Concatenate gaze and microexpression features
    features = np.concatenate((gaze_data, mexp_data), axis=1).reshape(1, -1)

    # Use a pre-trained SimpleImputer or ensure it is fitted with the training data
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)  # It's better to fit this with training data only

    # Predict using the SVM model
    prediction = svm_model.predict(features)
    print(prediction)
    
    # extract top 5 facial action units
    au_output = top5_au(mexp_filepath)
    facial_au_list = ""
    for i in range(len(au_output)-1):
        facial_au_list = facial_au_list + au_output[i] + ", "
    facial_au_list += au_output[-1]
    
    # Return the result
    if prediction == 'Deceptive': return 'DECEPTIVE', facial_au_list
    return 'TRUTHFUL', facial_au_list

############################################ FAU Functions #############################################################

def check_au_presence(row, au_columns):
    present_aus = []
    for au in au_columns:
        if row[au] == 1: present_aus.append(au)
    return present_aus

def top5_au(path):
    dic = {' AU04_c': 'Brow Lowerer',
           ' AU12_c': 'Lip Corner Puller',
           ' AU14_c': 'Dimpler',
           ' AU10_c': 'Upper Lip Raiser',
           ' AU05_c': 'Upper Lid Raiser',
           ' AU07_c': 'Lid Tightener',
           ' AU06_c': 'Cheek Raiser',
           ' AU23_c': 'Lip Tightener',
           ' AU17_c': 'Chin Raiser',
           ' AU15_c': 'Lip Corner Depressor',
           ' AU45_c': 'Blink',
           ' AU02_c': 'Outer Brow Raiser',
           ' AU25_c': 'Lips part',
           ' AU20_c': 'Lip stretcher',
           ' AU01_c': 'Inner Brow Raiser',
           ' AU26_c': 'Jaw Drop',
           ' AU09_c': 'Nose Wrinkler',
           ' AU28_c': 'Lip Suck'}

    au_columns = [' AU04_c', ' AU12_c', ' AU14_c', ' AU10_c', ' AU05_c', ' AU07_c', ' AU06_c', ' AU23_c', ' AU17_c',
                  ' AU15_c', ' AU45_c', ' AU02_c', ' AU25_c', ' AU20_c', ' AU01_c', ' AU26_c', ' AU09_c', ' AU28_c']
    au_counts = {au: 0 for au in au_columns}

    data = pd.read_csv(path)
    for index, row in data.iterrows():
        present_aus = check_au_presence(row, au_columns)
        for au in present_aus:
            au_counts[au] += 1

    sorted_au_counts = sorted(au_counts.items(), key=lambda item: item[1], reverse=True)
    top5_aus = [dic[au] for au, count in sorted_au_counts[:5]]

    return top5_aus

######################################### User Interface (UI) ##########################################################

with gr.Blocks() as mcs4ui:

    gr.Markdown(""" # MCS4 - Securing Face ID """)
    
    with gr.Tab("Home"):
        with gr.Row():
            with gr.Column(): 
                gr.Markdown("# Start Detecting Deceptions With Uploading A Video")
                gr.Markdown("""
                Deception can manifest in various forms within society. It can be broadly classified into two categories:

                **High-Risk Environments**: Situations where deception can lead to significant consequences. These include criminal investigations, national security matters, corporate fraud, and any scenario where truthfulness is critical to the outcome. In these high-stakes settings, the ability to accurately detect deception can be crucial for making informed decisions, ensuring justice, and maintaining security.

                **Incidental Deception**: Everyday situations where deception may occur with lesser impact. This includes social interactions, personal relationships, and minor disputes. Although the consequences of deception in these contexts are generally less severe, understanding and identifying deceptive behavior can still play a vital role in improving communication, trust, and overall social dynamics.

                Our advanced deception detection system leverages cutting-edge technology to analyze microexpressions and gaze patterns. By integrating data from these two modalities, our system can provide a comprehensive assessment of an individual's veracity. This multi-faceted approach ensures higher accuracy and reliability in detecting deceptive behaviors.

                ### Key Features:
                **Microexpression Analysis**: Microexpressions are brief, involuntary facial expressions that reveal genuine emotions. Our system is trained to recognize these subtle cues, which are often missed by the human eye.
                **Gaze Tracking**: Eye movement patterns can offer significant insights into a person's cognitive processes and truthfulness. By monitoring and analyzing gaze direction, fixation points, and blink rates, our system adds another layer of scrutiny to the deception detection process.
                **Machine Learning Integration**: Our system employs advanced machine learning algorithms, including support vector machines (SVM), to continuously improve detection accuracy. The model is trained on a diverse dataset to ensure robustness across various scenarios.
                **User-Friendly Interface**: Upload a video to start detecting deceptions effortlessly. The intuitive design allows users to navigate and utilize the system with ease, making it accessible for both professionals and non-experts alike.

                Our goal is to provide a reliable tool that can assist in identifying deceptive behaviors across different contexts, enhancing decision-making processes, and fostering an environment of trust and transparency.
                """)
            with gr.Column():
                gr.Image("user_interface/images/t1.jpeg", width=800, height=800)

    with gr.Tab("About"):
        gr.Markdown("""
                    
        ## About Us
        We are a team of undergraduate Computer Science students, MCS4 working on this project as part of our final year projects. 
        Our goal is to explore the complexities and applications of deception systems within digital environments, pushing the 
        boundaries of what's possible with current technology and contributing to academic understanding in this field.
        
        ## About the Deception Detection System
        Deception detection refers to the process of identifying and distinguishing between truthfulness and deception. It involves assessing and analyzing various verbal and non-verbal cues, macro and micro facial expressions, and body language. 
        In this project, we focus primarily on facial expressions, using features such as eyebrow movements, lips, eyes, nose, cheeks, and chin. These features can be extracted and classified into two groups: upper face action units and lower face action units (Torre et al., 2015). When analyzed correctly, these facial action units provide great detail in determining a person’s true emotion.

        ### Detailed Analysis of Facial Action Units:
        - **Upper Face Action Units**: This includes movements of the eyebrows, forehead, and upper eyelids. 
        - **Lower Face Action Units**: This involves the movements of the lower eyelids, cheeks, lips, and chin.

        ### Integration of Multimodal:
        - **Microexpression Analysis**: Microexpressions are brief, involuntary facial expressions that reveal genuine emotions. 
        - **Gaze Tracking**: Eye movement patterns offer significant insights into a person’s cognitive processes and truthfulness.
                    
        ### Ethical Considerations
        Please use this system responsibly. It is designed for educational purposes and should not be used to create misleading content that could harm individuals or entities.

        Deception detection technology, while powerful, carries significant ethical implications. The following points outline important considerations to ensure the responsible use of our system:

        - **Respect for Privacy**: The use of this system should always respect the privacy and consent of individuals being analyzed. Unauthorized use or covert recording without explicit consent is unethical and potentially illegal.
        - **Accuracy and Misinterpretation**: While our system is designed to be as accurate as possible, no technology is infallible. There is always a risk of false positives or false negatives. Users should be cautious in interpreting results and avoid making definitive judgments solely based on the system’s output.
        - **Purpose and Context**: The system is intended for educational and research purposes. It should not be used as a sole determinant in critical decisions, such as legal judgments, hiring processes, or personal relationships. Context is crucial, and the technology should be one of many tools used in a comprehensive evaluation process.
        - **Avoiding Harm**: The misuse of deception detection technology can lead to significant harm, including wrongful accusations, invasion of privacy, and erosion of trust. Users must consider the potential consequences of their actions and strive to minimize any negative impact.
        - **Transparency and Accountability**: Users of the system should be transparent about their use of the technology and be accountable for its application. This includes clearly communicating the limitations and intended use of the system to all stakeholders involved.
        - **Bias and Fairness**: Efforts should be made to ensure that the system is free from biases that could disproportionately affect certain groups. This involves continuous evaluation and improvement of the algorithms to address any detected biases.
        - **Ethical Training and Awareness**: Users should be educated about the ethical implications of deception detection technology. This includes understanding the broader societal impacts and the importance of ethical conduct in their application.

        By adhering to these ethical principles, users can help ensure that the deployment of deception detection technology is responsible, fair, and beneficial to society. Our goal is to foster a culture of ethical awareness and integrity in the use of advanced technological tools.
        """)

    with gr.Tab("Deception"):
        
        with gr.Tab("Video Analysis"):
            
            def video_identify(video):
                # generate csv file (mexp & gaze)
                #gaze_file = r"D:\\fit3162\\dataset\\output_gaze\\Gaze_reallifedeception_trial_lie_042.csv"
                #mexp_file = r"D:\\fit3162\\dataset\\output_micro_expression\\Mexp_reallifedeception_trial_lie_042.csv"
                result = predict_inp(video, get_model())
                return result
        
            def ui(video):
                if video is None: return "Please upload a file."
                elif not video.lower().endswith('.mp4'): 
                    
                    if video.lower().endswith('.mov'): 
                        video_new_path = convert_mov_to_mp4(video)
                        return video_identify(video_new_path)
                    else: return "Please upload a file with file type MP4 strictly"
                    
                else: return video_identify(video)

            combined_ui = gr.Interface(
                    fn=ui,
                    inputs=gr.Video(),
                    outputs=["text", "text"],
                    title="Deception Detection System",
                    description="Displays the result of the input video to identify authenticity."
            )
        
        gr.Markdown("""       
            Steps: 
            
            1. Select different tab of Real-time Camera Analysis (unavailable) or Video Analysis.
            2. Select a video or upload a video at the video input.
            3. Click Submit button.
            4. Wait for the result.
            5. The result will appear in output section indicating truthfulness.
            """)

    with gr.Tab("Dataset"):
                    
        gr.Interface(
            fn = count_gender_distribution,
            inputs=None,
            outputs="plot",
            title="Gender Bias Distribution"
        )
        
        gr.Markdown("""
            The graph has showed the overall percentages of gender bias to train the model.
            It has showed the gender bias distribution to allow users to visualise the 
            gender distribution (Latest updated: 13/5/2024)
                    """)

        gr.Interface(
            fn=count_training_distribution,
            inputs=None,
            outputs="plot",
            title="Training Data Distribution",
        )
        
        gr.Markdown("""
            The graph has showed the overall percentages of deceptive and truthful data to train the model.
            It has showed the label distribution to allow users to visualise the 
            percentages of real and deceptive distribution (Latest update: 13/5/2024)
                    """)

        gr.Interface(
            fn=count_dataset,
            inputs=None,
            outputs="text",
            title="Total data trained to build the model"
        )
        
        gr.Interface(
            fn=compute_accuracy,
            inputs=None,
            outputs="text",
            title="Overall Deceptive Detection System Accuracy "
        )
        
    
    with gr.Tab("Contact"):
        gr.Markdown("# The Team")
        
        gr.Image("user_interface/images/t2.jpeg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Shannon Theng\n"
                    "- **Role:** Product Manager\n"
                    "- **Email:** sthe0012@student.monash.edu")
        
        gr.Image("user_interface/images/t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jiahui Yin\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** jyin0021@student.monash.edu")
        
        gr.Image("user_interface/images/t3.jpeg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Kai Le Aw\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** kaww0003@student.monash.edu")
    
        gr.Image("user_interface/images/t4.jpeg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jing Wei Ong\n"
                    "- **Role:** Technical Lead\n"
                    "- **Email:** jong0074@student.monash.edu")
        
        gr.Image("user_interface/images/t5.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jessie Leong\n"
                    "- **Role:** Supervisor\n"
                    "- **Email:** leong.shumin@monash.edu\n"
                    "- **Phone:** +603-5516 1892")
    
if __name__ == '__main__':
    mcs4ui.launch()

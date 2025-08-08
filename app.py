import streamlit as st
import google.generativeai as genai
import requests
from io import BytesIO
import PyPDF2
import docx
from bs4 import BeautifulSoup
from google.cloud import speech
from google.oauth2 import service_account
import tempfile
import os
import json
import base64
from PIL import Image

# Page config
st.set_page_config(
    page_title="AI Playground - Gemini", 
    page_icon="🤖", 
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Simple authentication
def authenticate():
    st.title("🤖 AI Playground - Powered by Gemini")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Welcome! Enter any username to continue")
        username = st.text_input("Username", placeholder="Enter any name")
        
        if st.button("Enter Playground", use_container_width=True):
            if username:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Please enter a username")

# API Setup change ..............
def setup_apis():
    """Setup Gemini API and optional Google Cloud Speech from secrets"""
    if 'gemini_configured' not in st.session_state:
        st.sidebar.markdown("### 🔑 API Configuration")

        # --- Gemini API Key ---
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=gemini_key)
            
            # Test the key
            model = genai.GenerativeModel('gemini-1.5-flash')
            model.generate_content("Hello, testing API key")
            
            st.session_state.gemini_key = gemini_key
            st.session_state.gemini_configured = True
            st.sidebar.success("✅ Gemini API configured!")
        except KeyError:
            st.sidebar.error("❌ Gemini API key not found in secrets.toml")
            return False
        except Exception as e:
            st.sidebar.error(f"❌ Invalid Gemini API key: {str(e)}")
            return False

        # --- Google Cloud Speech ---
        try:
            credentials_json_string = st.secrets["GCP_CREDENTIALS"]
            credentials_info = json.loads(credentials_json_string) # Add this line
            st.session_state.gcp_credentials = service_account.Credentials.from_service_account_info(credentials_info)
            st.session_state.project_id = credentials_info.get('project_id')
            st.session_state.speech_configured = True
            st.sidebar.success("✅ Speech-to-Text configured!")
        except KeyError:
            st.sidebar.warning("⚠️ Google Cloud Speech credentials not found in secrets. Audio transcription will be disabled.")
        except Exception as e:
            st.sidebar.error(f"Error with Speech credentials: {str(e)}")
    

    return st.session_state.get('gemini_configured', False)

def analyze_image_gemini(image_bytes):
    """Analyze image using Gemini multimodal model"""
    try:
        # Convert image bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create comprehensive prompt
        prompt = """Analyze this image in detail and provide a comprehensive analysis including:

🔍 **Overall Scene & Setting:**
- What is the main subject/scene?
- Where does this appear to be taken?
- What's the overall mood or atmosphere?

📦 **Objects & Elements:**
- List and describe the main objects you can see
- Note their positions and relationships
- Identify any brands, text, or signage

👥 **People (if present):**
- Describe the people, their activities, clothing, expressions
- What are they doing?

🎨 **Visual Qualities:**
- Colors, lighting, composition
- Style (photograph, artwork, screenshot, etc.)
- Quality and technical aspects

📝 **Text & Writing:**
- Any text visible in the image?
- Signs, labels, captions, etc.

🤔 **Context & Interpretation:**
- What story does this image tell?
- What might be the purpose or context?
- Any notable or interesting details?

Organize your response with clear headings and be thorough but concise."""

        # Generate analysis
        response = model.generate_content([prompt, image])
        
        return response.text
        
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}\n\nPlease check your Gemini API key is valid and has sufficient quota."

def transcribe_audio_google(audio_bytes, filename):
    """Transcribe audio using Google Speech-to-Text, then analyze with Gemini"""
    
    # Check if Speech-to-Text is configured
    if not st.session_state.get('speech_configured'):
        return {
            "error": "Google Cloud Speech-to-Text not configured. Please upload your service account JSON in the sidebar for audio transcription functionality."
        }
    
    try:
        # Initialize Speech client
        client = speech.SpeechClient(credentials=st.session_state.gcp_credentials)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Read the audio file
            with open(temp_file_path, 'rb') as audio_file:
                audio_content = audio_file.read()
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
        
        # Auto-detect audio format
        def get_audio_encoding(filename):
            ext = filename.lower().split('.')[-1]
            encoding_map = {
                'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
                'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
                'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
                'ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
                'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            }
            return encoding_map.get(ext, speech.RecognitionConfig.AudioEncoding.LINEAR16)
        
        # Initialize variables at the start
        full_transcript = ""
        diarized_text = ""
        speakers_detected = 0
        duration = 0
        
        # Try with speaker diarization first
        try:
            # Configure recognition with proper diarization setup
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Create diarization config separately (newer API format)
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=2,
            )
            
            config = speech.RecognitionConfig(
                encoding=get_audio_encoding(filename),
                sample_rate_hertz=16000,
                language_code="en-US",
                diarization_config=diarization_config,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
            )
            
            # Perform the transcription
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=300)
            
            # Process results with speaker information
            full_transcript = ""
            speaker_segments = []
            
            # Extract speaker-separated segments
            for result in response.results:
                for alternative in result.alternatives:
                    full_transcript += alternative.transcript + " "
                    
                    # Process words with speaker tags
                    for word_info in alternative.words:
                        speaker_segments.append({
                            'word': word_info.word,
                            'speaker': word_info.speaker_tag,
                            'start_time': word_info.start_time.total_seconds(),
                            'end_time': word_info.end_time.total_seconds()
                        })
            
            # Create diarized transcript
            diarized_text = ""
            if speaker_segments:
                current_speaker = None
                current_segment = []
                
                for word_data in speaker_segments:
                    if word_data['speaker'] != current_speaker:
                        # New speaker - output previous segment
                        if current_segment:
                            speaker_label = f"Speaker {current_speaker}" if current_speaker else "Speaker Unknown"
                            diarized_text += f"\n**{speaker_label}:** {' '.join(current_segment)}\n"
                        
                        # Start new segment
                        current_speaker = word_data['speaker']
                        current_segment = [word_data['word']]
                    else:
                        current_segment.append(word_data['word'])
                
                # Add final segment
                if current_segment:
                    speaker_label = f"Speaker {current_speaker}" if current_speaker else "Speaker Unknown"
                    diarized_text += f"\n**{speaker_label}:** {' '.join(current_segment)}\n"
            
            # If no real speaker diarization, use Gemini for intelligent analysis
            if not diarized_text or diarized_text == "Speaker diarization unavailable - using AI analysis instead" or "Speaker Unknown" in diarized_text:
                st.info("🤖 Using Gemini AI for intelligent speaker analysis...")
                
                gemini_diarization_prompt = f"""This appears to be a conversation between two people. Analyze the transcript and intelligently separate it into speakers based on:

1. **Conversation flow** (questions vs answers)
2. **Topic changes** and **perspective shifts**
3. **Speaking styles** and **vocabulary differences**
4. **Context clues** (who asks vs who responds)

**Transcript to analyze:**
{full_transcript}

**Instructions:**
- Look for natural conversation patterns
- Identify where one person stops and another starts
- Pay attention to questions, responses, topic introductions
- Format as: **Speaker A:** [text] and **Speaker B:** [text]
- Be logical about speaker changes based on conversation flow

**Provide the intelligently diarized conversation:**"""

                try:
                    diarization_model = genai.GenerativeModel('gemini-1.5-flash')
                    diarization_response = diarization_model.generate_content(gemini_diarization_prompt)
                    gemini_diarized_text = diarization_response.text
                    
                    # Use Gemini's diarization as the primary result
                    diarized_text = f"""🤖 **AI-Enhanced Speaker Diarization:**

{gemini_diarized_text}

---
*Note: This diarization uses AI conversation analysis to identify likely speaker changes based on dialogue patterns, context, and speaking styles.*"""
                    
                except Exception as e:
                    diarized_text = f"AI diarization failed: {str(e)}"
            
        except Exception as diarization_error:
            # Fallback: Basic transcription without speaker diarization
            st.warning(f"Speaker diarization failed ({str(diarization_error)}). Using basic transcription + AI analysis.")
            
            # Simple config without diarization
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=get_audio_encoding(filename),
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
            
            # Perform basic transcription
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=300)
            
            # Extract text
            full_transcript = ""
            for result in response.results:
                for alternative in result.alternatives:
                    full_transcript += alternative.transcript + " "
            
            diarized_text = "Speaker diarization unavailable - using AI analysis instead"
            speakers_detected = "Unknown"
            duration = 0  # Reset duration for basic transcription
        
        # Now use Gemini to enhance the analysis
        gemini_analysis = ""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            analysis_prompt = f"""Analyze this conversation transcript and provide insights:

**Transcript:**
{full_transcript}

Please provide:

🗣️ **Speaker Analysis & Diarization:**
- Try to identify where different speakers are likely talking based on conversation flow
- What can you tell about each speaker? (speaking style, role, etc.)
- Suggest improved speaker separation if you can detect patterns

📊 **Conversation Analysis:**
- What is the main topic/theme?
- Key points discussed
- Tone and mood of the conversation
- Who seems to be leading the conversation?

📝 **Summary:**
- Brief summary of what was discussed
- Important conclusions or decisions

🔍 **Insights:**
- Interesting patterns or observations
- Communication style analysis
- Key takeaways

If you can detect likely speaker changes in the transcript, please reformat it with **Speaker A:** and **Speaker B:** labels based on conversation flow, questions/answers, and topic changes.

Format your response clearly with the headers shown above."""

            response = model.generate_content(analysis_prompt)
            gemini_analysis = response.text
            
        except Exception as e:
            gemini_analysis = f"Gemini analysis unavailable: {str(e)}"
        
        return {
            "full_transcription": full_transcript.strip(),
            "speaker_diarization": diarized_text if diarized_text else "No speaker diarization available",
            "gemini_analysis": gemini_analysis,
            "word_count": len(full_transcript.split()),
            "duration": f"{duration:.1f} seconds" if duration > 0 else "Unknown",
            "speakers_detected": speakers_detected
        }
        
    except Exception as e:
        return {"error": f"Error transcribing audio: {str(e)}"}

def extract_text_from_pdf(pdf_bytes):
    """Enhanced PDF text extraction with visual content analysis"""
    try:
        # Method 1: Try PyPDF2 for standard text
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        standard_text = ""
        
        for page in pdf_reader.pages:
            standard_text += page.extract_text() + "\n"
        
        # Method 2: Convert PDF pages to images and analyze with Gemini Vision
        try:
            import fitz  # PyMuPDF
            
            visual_analysis = ""
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(min(3, len(doc))):  # Process first 3 pages max
                page = doc.load_page(page_num)
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Analyze with Gemini Vision
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    image = Image.open(BytesIO(img_data))
                    
                    prompt = f"""Analyze this PDF page (page {page_num + 1}) and extract ALL visible content including:

📄 **Text Content:**
- All readable text, including styled text, headers, labels
- Menu items, prices, product names
- Contact information, addresses, phone numbers
- Any text that might be embedded in images or styled layouts

📊 **Structured Information:**
- Tables, lists, menus, pricing
- Categories and subcategories  
- Any organized data or information

🎨 **Visual Elements:**
- Descriptions of images, logos, graphics
- Layout structure and organization
- Visual hierarchy and formatting

Format your response to capture ALL information visible on this page, including text that might not be extractable by standard PDF text extraction."""

                    response = model.generate_content([prompt, image])
                    visual_analysis += f"\n--- PAGE {page_num + 1} VISUAL ANALYSIS ---\n"
                    visual_analysis += response.text + "\n"
                    
                except Exception as vision_error:
                    visual_analysis += f"\n--- PAGE {page_num + 1} VISUAL ANALYSIS FAILED ---\n"
                    visual_analysis += f"Error: {str(vision_error)}\n"
            
            doc.close()
            
            # Combine both methods
            if visual_analysis.strip():
                combined_text = f"""📄 STANDARD PDF TEXT EXTRACTION:
{standard_text}

🔍 VISUAL CONTENT ANALYSIS:
{visual_analysis}"""
                return combined_text
            else:
                return standard_text
                
        except ImportError:
            # Fallback: PyPDF2 only with warning
            return f"""⚠️ BASIC TEXT EXTRACTION ONLY:
{standard_text}

Note: For complete PDF analysis including visual elements, menus, and styled text, please install PyMuPDF:
pip install PyMuPDF

This would enable visual analysis of PDF pages to capture content that standard text extraction misses."""
            
        except Exception as visual_error:
            # Fallback to standard text with note
            return f"""📄 STANDARD PDF TEXT:
{standard_text}

⚠️ Visual analysis unavailable: {str(visual_error)}"""
            
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(docx_bytes):
    """Extract text from Word document"""
    try:
        doc = docx.Document(BytesIO(docx_bytes))
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        return text
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def scrape_url_content(url):
    """Extract text content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:15000]  # Increased limit for Gemini
        
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

def summarize_with_gemini(text, summary_type="detailed", content_type="document"):
    """Summarize text using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create context-aware prompts
        if summary_type == "detailed":
            prompt = f"""Provide a comprehensive summary of this {content_type}. Include:

📋 **Main Points:**
- Key themes and topics
- Important arguments or findings
- Major conclusions

🔍 **Detailed Analysis:**
- Supporting details and evidence
- Context and background
- Implications or significance

📊 **Structure:**
- How the content is organized
- Flow of ideas or narrative

💡 **Key Takeaways:**
- Most important insights
- Actionable information
- Notable quotes or statistics (if any)

**Content to summarize:**
{text}

Please organize your response with clear headings and bullet points where appropriate."""

        elif summary_type == "brief":
            prompt = f"""Provide a concise summary of this {content_type} in 3-4 sentences. Focus on:
- The main point or theme
- Key conclusion or takeaway
- Most important detail

**Content to summarize:**
{text}

Keep it brief but informative."""

        else:  # bullet_points
            prompt = f"""Summarize this {content_type} as organized bullet points:

• **Main Topic:** [What is this about?]
• **Key Points:** 
  - [First major point]
  - [Second major point]
  - [Third major point]
• **Important Details:**
  - [Notable detail 1]
  - [Notable detail 2]
• **Conclusion/Outcome:** [Final takeaway]

**Content to summarize:**
{text}

Format as clean, scannable bullet points."""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error creating summary: {str(e)}\n\nPlease check your Gemini API key and quota."

# Main Application
def main_app():
    # Sidebar
    st.sidebar.title(f"👋 Hello, {st.session_state.username}!")
    
    # Check API setup
    if not setup_apis():
        st.title("🤖 AI Playground - Powered by Gemini")
        st.markdown("### Welcome to your Gemini-powered AI Playground!")
        st.info("👈 Please add your Gemini API key in the sidebar to get started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 **What you can do:**
            - 🖼️ **Image Analysis** - Upload images for comprehensive AI analysis
            - 🎵 **Audio Transcription** - Convert speech to text with speaker insights
            - 📄 **Document Summarization** - Summarize PDFs, Word docs, and web pages
            """)
        
        with col2:
            st.markdown("""
            ### 🔑 **Getting Started:**
            1. Get your free Gemini API key from [aistudio.google.com](https://aistudio.google.com)
            2. Paste it in the sidebar
            3. Start using AI features!
            
            **Free Limits:** 15 requests/minute, 1,500/day
            """)
        return
    
    # Show API status
    st.sidebar.success("🤖 Gemini API: Connected")
    if st.session_state.get('speech_configured'):
        st.sidebar.success("🎙️ Speech-to-Text: Connected")
    else:
        st.sidebar.info("🎙️ Speech-to-Text: Not configured (audio features disabled)")
    
    # Sidebar logout
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.clear()
        st.rerun()
    
    # Main content
    st.title("🤖 AI Playground - Powered by Gemini")
    st.markdown("### Choose your AI-powered tool:")
    
    # Tool selection
    tool = st.selectbox(
        "Select a tool:",
        ["🖼️ Image Analysis (Gemini Vision)", "🎵 Audio Transcription + Analysis", "📄 Document & URL Summarization"]
    )
    
    st.markdown("---")
    
    # Image Analysis Tool
    if tool == "🖼️ Image Analysis (Gemini Vision)":
        st.header("🖼️ Gemini Vision Analysis")
        st.markdown("Upload an image for comprehensive AI analysis using Google's multimodal Gemini model")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Your Image")
                st.image(uploaded_file, use_column_width=True)
                
                # Image info
                st.info(f"**Filename:** {uploaded_file.name}\n**Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
            
            with col2:
                st.subheader("Gemini Analysis")
                
                if st.button("🔍 Analyze with Gemini Vision", use_container_width=True):
                    with st.spinner("Analyzing image with Gemini..."):
                        analysis = analyze_image_gemini(uploaded_file.getvalue())
                        st.success("Analysis complete!")
                        st.markdown(analysis)
    
    # Audio Transcription Tool
    elif tool == "🎵 Audio Transcription + Analysis":
        st.header("🎵 Audio Transcription with Gemini Analysis")
        st.markdown("Upload audio for professional transcription + AI-powered conversation analysis")
        
        if not st.session_state.get('speech_configured'):
                st.markdown("**Alternative:** Upload your audio to a transcription service like [otter.ai](https://otter.ai) or [rev.com](https://rev.com), then paste the transcript here for AI analysis:")
                
                transcript_text = st.text_area(
                    "Or paste a conversation transcript here:",
                    height=200,
                    placeholder="Speaker A: Hello, how are you?\nSpeaker B: I'm doing well, thanks for asking...\n\nOr just paste the raw transcript and let Gemini figure out the speakers!"
                )
                
                if transcript_text and st.button("🤖 Analyze Transcript with Gemini"):
                    with st.spinner("Analyzing conversation with Gemini..."):
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        comprehensive_analysis_prompt = f"""Analyze this conversation transcript comprehensively:

**Transcript:**
{transcript_text}

Provide a complete analysis with:

🗣️ **Intelligent Speaker Diarization:**
- Separate the conversation into speakers based on dialogue patterns
- Format as **Speaker A:** and **Speaker B:** 
- Explain your reasoning for speaker separation

📊 **Conversation Analysis:**
- Main topics and themes discussed
- Key insights and conclusions
- Communication dynamics and flow
- Who leads the conversation and how

🎯 **Speaker Profiles:**
- Speaking style of each person
- Role in the conversation (interviewer/interviewee, friends, business, etc.)
- Personality indicators from speech patterns

📝 **Detailed Summary:**
- What was discussed in detail
- Important points and decisions
- Outcome or resolution

🔍 **Advanced Insights:**
- Emotional tone and mood
- Power dynamics or relationship type
- Communication effectiveness
- Notable patterns or interesting observations

Format your response with clear sections and make it comprehensive."""

                        try:
                            response = model.generate_content(comprehensive_analysis_prompt)
                            st.success("Comprehensive Analysis Complete!")
                            st.markdown(response.text)
                            
                            # Add word count
                            word_count = len(transcript_text.split())
                            st.info(f"📊 **Analysis Stats:** {word_count} words processed")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                st.markdown("---")
        else:
            uploaded_file = st.file_uploader(
                "Choose an audio file...", 
                type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm']
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Your Audio")
                    st.audio(uploaded_file)
                    
                    # Audio info
                    st.info(f"**Filename:** {uploaded_file.name}\n**Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
                    
                    st.warning("⏱️ Processing may take 1-3 minutes for longer audio files")
                
                with col2:
                    st.subheader("Transcription + Analysis")
                    
                    if st.button("🎯 Transcribe & Analyze", use_container_width=True):
                        with st.spinner("Transcribing and analyzing with Gemini..."):
                            result = transcribe_audio_google(uploaded_file.getvalue(), uploaded_file.name)
                            
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.success("Analysis complete!")
                                
                                # Tabs for different views
                                tab1, tab2, tab3, tab4 = st.tabs(["📝 Transcript", "👥 Speakers", "🤖 Gemini Analysis", "📊 Stats"])
                                
                                with tab1:
                                    st.markdown("**Full Transcription:**")
                                    st.write(result["full_transcription"])
                                
                                with tab2:
                                    st.markdown("**Speaker Diarization:**")
                                    st.markdown(result["speaker_diarization"])
                                
                                with tab3:
                                    st.markdown("**Gemini Conversation Analysis:**")
                                    st.markdown(result["gemini_analysis"])
                                
                                with tab4:
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    with col_stat1:
                                        st.metric("Word Count", result['word_count'])
                                    with col_stat2:
                                        st.metric("Duration", result['duration'])
                                    with col_stat3:
                                        st.metric("Speakers", result.get('speakers_detected', 'N/A'))
    
    # Document Summarization Tool
    else:  # Document Summarization
        st.header("📄 Gemini-Powered Document & URL Summarization")
        st.markdown("Upload documents or provide URLs for intelligent AI summaries using Gemini's advanced language understanding")
        
        # Choose input type
        input_type = st.radio(
            "Choose input type:",
            ["📄 Upload Document", "🌐 Enter URL"]
        )
        
        if input_type == "📄 Upload Document":
            uploaded_file = st.file_uploader(
                "Choose a document...", 
                type=['pdf', 'docx', 'txt']
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Document Info")
                    st.info(f"**Filename:** {uploaded_file.name}\n**Type:** {uploaded_file.type}\n**Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
                    
                    # Summary type
                    summary_type = st.selectbox(
                        "Summary type:",
                        ["detailed", "brief", "bullet_points"],
                        help="Choose how detailed you want the summary"
                    )
                
                with col2:
                    st.subheader("Gemini Summary")
                    
                    if st.button("📝 Generate Enhanced Summary with Gemini", use_container_width=True):
                        with st.spinner("Extracting text (including visual content) and generating summary with Gemini..."):
                            # Extract text based on file type
                            if uploaded_file.type == "application/pdf":
                                text = extract_text_from_pdf(uploaded_file.getvalue())
                            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                text = extract_text_from_docx(uploaded_file.getvalue())
                            else:  # txt
                                text = uploaded_file.getvalue().decode("utf-8")
                            
                            if text.startswith("Error"):
                                st.error(text)
                            else:
                                # Generate summary with Gemini
                                summary = summarize_with_gemini(text, summary_type, "document")
                                st.success("Summary generated by Gemini!")
                                st.markdown(summary)
                                
                                # Show text preview
                                with st.expander("📖 View extracted text (first 500 chars)"):
                                    st.text(text[:500] + "..." if len(text) > 500 else text)
        
        else:  # URL input
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("URL Input")
                url = st.text_input("Enter URL:", placeholder="https://example.com/article")
                
                # Summary type
                summary_type = st.selectbox(
                    "Summary type:",
                    ["detailed", "brief", "bullet_points"],
                    help="Choose how detailed you want the summary"
                )
            
            with col2:
                st.subheader("Gemini Summary")
                
                if st.button("🌐 Summarize URL with Gemini", use_container_width=True):
                    if url:
                        with st.spinner("Scraping content and generating summary with Gemini..."):
                            # Scrape content
                            text = scrape_url_content(url)
                            
                            if text.startswith("Error"):
                                st.error(text)
                            else:
                                # Generate summary with Gemini
                                summary = summarize_with_gemini(text, summary_type, "web page")
                                st.success("Summary generated by Gemini!")
                                st.markdown(summary)
                                
                                # Show content preview
                                with st.expander("📖 View scraped content (first 500 chars)"):
                                    st.text(text[:500] + "..." if len(text) > 500 else text)
                    else:
                        st.error("Please enter a URL")

# App entry point
def main():
    if not st.session_state.authenticated:
        authenticate()
    else:
        main_app()

if __name__ == "__main__":
    main()

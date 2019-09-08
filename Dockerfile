FROM tensorflow/tensorflow

COPY ./ /

# Install python package manager, pyaudio and device manager packages
RUN apt update -y && apt install -y python3-pip python3-pyaudio alsa-base alsa-utils

# Install dependencies for playsound
RUN apt install -y python-gi python3-gi \
    gstreamer1.0-tools \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

# Install google calendar API dependencies
RUN pip3 install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
    
# Install python modules
RUN pip3 install playsound SpeechRecognition gTTs nltk numpy tflearn tensorflow python-dateutil
RUN python3 -c "import nltk; nltk.download('punkt')"

# Train jarvis
RUN mkdir -p /model
RUN python3 /trainJarvis.py

# Change default speaker
RUN sed -i 's/".pcm.default:CARD=" \$CARD/".pcm.default:CARD=" 1/g' /usr/share/alsa/pcm/default.conf

# Execute jarvis at startup
CMD ["python3", "/jarvis.py"]
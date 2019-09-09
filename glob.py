# Data structures used by our neural network
words = []
labels = []
training = []
output = []
model = []
data = []

# Context of the current and old conversation
context = "startup"
new_context = "intents"

# Language used by gTTS
lang = None

# Google calendar API object
service = None
import gtts.lang
from gtts import gTTS
import speech_recognition as sr
import os
#print(gtts.lang.tts_langs())
print("Press a for Speech to Text")
print("Press b for Text to Speech")
option = input("Enter your options: ")
def speech_to_text():
    voice = sr.Recognizer()
    with sr.Microphone() as source:
        voice.adjust_for_ambient_noise(source)
        print("Please speak !!!!!")
        audio = voice.listen(source)
    try:
        outfile = open("output_in_text.txt", "w")
        outtext = voice.recognize_google(audio)
        outfile.write(outtext)
        print("The words told to api are: " + outtext)
    except:
        print("Please use English you prha likha jahil !!!!!")

def text_to_speech():
    print("Press a for entering text from a text.txt")
    print("Press b for entering text from output panel")
    choice = input("Enter your choice (should be a or b): ")
    if choice == 'a':
        rec_text_open = open("recorded_text.txt", "r")
        rec_text_replace = rec_text_open.read().replace('\n', " ")
        rec_text_open.close()
        converted_output = gTTS(text=rec_text_replace, lang="en", slow=False)
        converted_output.save("recorded_text.mp3")
        os.system("recorded_text.mp3")
    elif choice == 'b':
        enter_your_text = input("Please Enter your Text: ")
        texted_output = gTTS(text=enter_your_text, lang="en", slow=False)
        texted_output.save("typed_text.mp3")
        os.system("typed_text.mp3")

if option == 'a':
    speech_to_text()
if option == 'b':
    text_to_speech()
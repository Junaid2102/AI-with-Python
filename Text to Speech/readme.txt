In this application you can convert text into speech and speech into text.

						====== For Speech to text ======

At first we got voice from microphone and then we stored that voice and then that voice is converted into text with help of .recognize and that text is stored in a text file.
You can change the name of text file on your own.
We did error handling so user must enter only in english, if you speak any other language than english application will show error message.

						====== For Text to Speech ======

User will choose what he want whether he want to enter text from console or from a text file.
If user choose input method from console then provided input would be directly provided to google translation library and that output would be saved in .mp3 file and would be opened by calling os.system.
If user choose input method from a text file then our api will get text from file and provide that text to google translation library and will get .mp3 file and would be opened by calling os.system.

						====== Thank You ======
						====== Junaid Alam ======
import os
from tkinter import Tk, X , Frame, Button, Label, PhotoImage, Listbox, Scrollbar, END, ACTIVE, SINGLE, StringVar,Toplevel
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import logging
logging.disable(logging.WARNING)
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import time
import random
import pygame
from pygame import mixer
global max_gesture 


def facial_emotion_detection():
    model = model_from_json(open("static\Final_jsonmodel.json", "r").read())
    model.load_weights('static\Final.h5')

    face_haar_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)

    emotion_list = []
    cv2.namedWindow('Facial Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Facial Emotion Detection', 640, 480)  # Adjust window size

    while len(emotion_list) < 30:
        res, frame = cap.read()

        height, width , channel = frame.shape
        sub_img = frame[0:int(height/6), 0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            emotion_list.append(emotion_prediction)

            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.7
            FONT_THICKNESS = 2
            lable_color = (10, 10, 255)
            lable_violation = 'Sentiment: {}'.format(emotion_prediction)

            # Apply horizontal flip to the frame (mirror view)
            frame_flipped = cv2.flip(frame, 1)

            # Calculate text position based on the original frame size
            violation_text_dimension = cv2.getTextSize(lable_violation, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            violation_x_axis = int((width - violation_text_dimension[0]) - 10)
            cv2.putText(frame_flipped, lable_violation, (violation_x_axis, int(height/6) - 10), FONT, FONT_SCALE, lable_color, FONT_THICKNESS)

            # Display the mirrored view
            cv2.imshow('Facial Emotion Detection', frame_flipped)
            cv2.setWindowProperty('Facial Emotion Detection', cv2.WND_PROP_TOPMOST, 1)  # Make window stay on top
            cv2.waitKey(60)  # Capture for 60 milliseconds

    cap.release()
    cv2.destroyAllWindows()

    # Count occurrences of each emotion
    emotion_counts = {emotion: emotion_list.count(emotion) for emotion in set(emotion_list)}

    # Find the emotion with the maximum count
    max_gesture = max(emotion_counts, key=emotion_counts.get)
    print(emotion_list)
    if max_gesture != 'neutral':
        print("Most detected emotion:", max_gesture)
        play_random_song(max_gesture)
    else:
        # Check the second most detected emotion
        del emotion_counts[max_gesture]  # Remove 'neutral' from counts
        second_max_gesture = max(emotion_counts, key=emotion_counts.get)

        # Check if the count of the second max emotion is greater than or equal to 9
        if emotion_counts[second_max_gesture] >= 9:
            print("Second most detected emotion:", second_max_gesture)
            play_random_song(second_max_gesture)
            second_max_gesture = max_gesture
        else:
            print("Try Hand Gesture")

gesture_emotion_map = {"peace ":"sad","live long":"sad","thumbs down ": "disgust","stop ":"fear","rock ": "surprise","call me":"surprise","fist ":"angry","smile ":"happy","okay ":"happy","thumbs up ":"happy"}
        

def hand_gesture_detection():
    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = load_model('mp_hand_gesture')

    # Load class names
    with open('gesture.names', 'r') as f:
        classNames = f.read().split('\n')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    

    gesture_history = []  # Variable to store detected gestures 

    last_detection_time = time.time()

    cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Gesture Detection', 640, 480)  # Adjust window size 

    while True:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        className = ''

        # post-process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

                # Store detected gesture in variable
                gesture_history.append(className)

                last_detection_time = time.time()

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

        # Show the webcam view
        cv2.imshow("Hand Gesture Detection", frame)
        cv2.setWindowProperty('Hand Gesture Detection', cv2.WND_PROP_TOPMOST, 1)  # Make window stay on top

        # Automatically close after storing the gesture
        if time.time() - last_detection_time >= 5:
            break
        if len(gesture_history) >= 30:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    # Access gesture_history variable to see detected gestures
    print("Detected Gestures:", gesture_history)

    # Find the most common gesture in gesture_history (All Brain cells died here!!)
    max_gesture = max(set(gesture_history), key=gesture_history.count)
    
    a = gesture_emotion_map[max_gesture]
    print("Folder :- ",a)
    print("Max Gesture:", max_gesture)
    play_random_song(a)
    

def play_random_song(folder_path):
    top = Toplevel()
    top.geometry("300x200")
    top.wm_attributes("-topmost", 1)
    top.rowconfigure(0, weight=1)
    top.columnconfigure(0, weight=1)
    height = 500
    width = 600
    x = (top.winfo_screenwidth()//2)-(width//2)
    y = (top.winfo_screenheight()//4)-(height//4)
    top.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    window  =  Frame(top)
    window2 = Frame(top)

    for frame in (window, window2):
        frame.grid(row=0, column=0, sticky='nsew')

    def show_frame(frame):
        frame.tkraise()

    show_frame(window)

    window.config(background='#7852E6')

    # Get the directory of the script
    script_dir = os.path.dirname(__file__)

    # Construct relative paths for images folder
    images_dir = os.path.join(script_dir, "images")

    image_image_4 = PhotoImage(file=os.path.join(images_dir, "image_4.png"))
    image_4 = Label(window, bg='#7852E6', image=image_image_4)
    image_4.place(x=41, y=41)

    image_image_1 = PhotoImage(file=os.path.join(images_dir, "image_1.png"))
    image_1 = Label(window, bg='#7852E6', image=image_image_1)
    image_1.place(x=196, y=121)

    image_image_2 = PhotoImage(file=os.path.join(images_dir, "image_2.png"))
    image_2 = Label(window, bg='#7852E6', image=image_image_2)
    image_2.place(x=60, y=340)

    button_image_2 = PhotoImage(file=os.path.join(images_dir, "button_2.png"))
    button_2 = Button(window, image=button_image_2, borderwidth=0, highlightthickness=0, command=lambda: play(), relief="flat", activebackground='#7852E6')
    button_2.place(x=243, y=386.0, width=80.0, height=81.0)

    button_image_3 = PhotoImage(file=os.path.join(images_dir, "button_3.png"))
    button_3 = Button(window, image=button_image_3, borderwidth=0, highlightthickness=0, command=lambda: next_song(), relief="flat", activebackground='#7852E6')
    button_3.place(x=355, y=397.0, width=74.0, height=56.0)

    button_image_4 = PhotoImage(file=os.path.join(images_dir, "button_4.png"))
    button_4 = Button(window, image=button_image_4, borderwidth=0, highlightthickness=0, command=lambda: previous_song(), relief="flat", activebackground='#7852E6')
    button_4.place(x=137, y=398.0, width=74.0, height=56.0)

    image_image_3 = PhotoImage(file=os.path.join(images_dir, "image_3.png"))
    image_3 = Label(window, bg='#000000', image=image_image_3)
    image_3.place(x=203, y=131)

    button_image_5 = PhotoImage(file=os.path.join(images_dir, "button_5.png"))
    button_5 = Button(window, image=button_image_5, borderwidth=0, highlightthickness=0, command=lambda: show_frame(window2), relief="flat", activebackground='#7852E6')
    button_5.place(x=49, y=378.0, width=76.0, height=76.0)

    playing_song = Label(window, text="", bg='#7852E6', fg='#ffffff', font=('yu gothic ui', 8, 'bold'))
    playing_song.place(x=150, y=360, height=20, width=300)

    window2.config(background='#7852E6')

    button_image_6 = PhotoImage(file=os.path.join(images_dir, "button_2.png"))
    button_6 = Button(window2, image=button_image_6, borderwidth=0, highlightthickness=0, command=lambda: call_play(), relief="flat", activebackground='#7852E6')
    button_6.place(x=100.0, y=2.0, width=80.0, height=80.0)

    def call_play():
        show_frame(window)
        play()

    button_7 = Button(window2, text="""BACK""", borderwidth=0, highlightthickness=0, command=lambda: show_frame(window), relief="flat", activebackground='#000000', bg='#000000', fg='#ffffff')
    button_7.place(x=26.0, y=25.0, width=54.0, height=33.0)

    listbox = Listbox(window2, selectmode=SINGLE, bg='#000000', fg='#ffffff', font=('yu gothic ui', 10, 'bold'), bd=25, relief='flat')
    listbox.place(x=30, y=80, height=406, width=520)

    scroll = Scrollbar(window2)
    scroll.place(x=550, y=80, height=406)

    listbox.config(yscrollcommand=scroll.set)
    scroll.config(command=listbox.yview)

# Set the working directory to the directory of the script
    os.chdir(script_dir)

# Change directory to the "songs" folder
    os.chdir(folder_path)

    songs = os.listdir()

    button_image_1 = PhotoImage(file=os.path.join(images_dir, "button_1.png"))

    def play():
        current_song = listbox.get(ACTIVE)
        playing_song['text'] = current_song
        mixer.music.load(current_song)
        mixer.music.play()

        button_1 = Button(window, image=button_image_1, borderwidth=0, highlightthickness=0, command=lambda: pause_song(), relief="flat", activebackground='#7852E6')
        button_1.place(x=241, y=385.0, width=80.0, height=81.0)

    resume_pic = PhotoImage(file=os.path.join(images_dir, "button_2.png"))

    def pause_song():
        mixer.music.pause()

        resume_button = Button(window, image=resume_pic, borderwidth=0, highlightthickness=0, command=lambda: resume_song(), relief="flat", activebackground='#7852E6')
        resume_button.place(x=243, y=386.0, width=80.0, height=81.0)

    def resume_song():
      mixer.music.unpause()
      button_1 = Button(window, image=button_image_1, borderwidth=0, highlightthickness=0, command=lambda: pause_song(), relief="flat", activebackground='#7852E6')
      button_1.place(x=241, y=385.0, width=80.0, height=81.0)

    def next_song():
        playing = playing_song['text']
        index = songs.index(playing)
        next_index = index + 1
        playing = songs[next_index]
        mixer.music.load(playing)
        mixer.music.play()
        listbox.delete(0, END)
        song_list()
        listbox.select_set(next_index)
        playing_song['text'] = playing

    def previous_song():
        playing = playing_song['text']
        index = songs.index(playing)
        next_index = index - 1
        playing = songs[next_index]
        mixer.music.load(playing)
        mixer.music.play()
        listbox.delete(0, END)
        song_list()
        listbox.select_set(next_index)
        playing_song['text'] = playing

    def song_list():
        for i in songs:
            listbox.insert(END, i)

    song_list()



    mixer.init()
    songs_state = StringVar()
    top.resizable(False, False)
    top.mainloop()   
    






def main():
    
    # choice = input("Choose detection type \n 1: Facial Emotion Detection \n 2: Hand Gesture Detection\nYour choice: ")
    # if choice == '1':
    #     facial_emotion_detection()
    # elif choice == '2':
    #     hand_gesture_detection()
    # else:
    #     print("Invalid choice")
    
  


    root = Tk()
    root.geometry("300x200")  # Increased window size to accommodate the buttons
    root.rowconfigure(0, weight=1)
    root.config(background='#7852E6')
    root.columnconfigure(0, weight=1,)
    height = 500
    width = 600
    x = (root.winfo_screenwidth()//2)-(width//2)
    y = (root.winfo_screenheight()//4)-(height//4)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    window = Frame(root)
    window.config(background='#7852E6')
     # Get the directory of the script
    script_dir = os.path.dirname(__file__)

    # Construct relative paths for images folder
    images_dir = os.path.join(script_dir, "images")
    # Load background image
    image_image_5 = PhotoImage(file=os.path.join(images_dir, "image_5.png"))
    image_5 = Label(root, bg='#7852E6', image=image_image_5)
    image_5.place(x=0, y=0, relwidth=1, relheight=1) # Set the label to occupy full window size

    # Create frame for buttons
    window = Frame(root)
    window.place(relx=0.5, rely=0.5,)  # Center the frame

    # Load button images
    b1_image = PhotoImage(file=os.path.join(images_dir, "b1.png"))
    b2_image = PhotoImage(file=os.path.join(images_dir, "b2.png"))

    
    
    # Create buttons
    b1 = Button(window, image=b1_image, borderwidth=0, highlightthickness=0, command=lambda: facial_emotion_detection(), relief="flat", activebackground='#FFFFFF', bg=root.cget('bg') )
    # b1.config(bg=root.cget('bg'))
    # b1.pack(side="left", anchor='center')
    b1.grid(row=0, column=0, ipadx=5)
    
    

    b2 = Button(window, image=b2_image, borderwidth=0, highlightthickness=0, command=lambda: hand_gesture_detection(), relief="flat", activebackground='#FFFFFF', bg=root.cget('bg'))
    # b2.config(bg=root.cget('bg'))  
    # b2.pack(side="left", anchor='center')
    b2.grid(row=0, column=2, ipadx=5)
    
    

    root.mainloop()

if __name__ == "__main__":
    main()

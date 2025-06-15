from tkinter import *
from PIL import Image, ImageTk, ImageSequence
from astra_gui_final import *

# ------------------- LOGIN PAGE -------------------
class LoginPage:
    def __init__(self, window):
        self.window = window
        self.window.geometry('1166x718')

        self.window.title('AstraBharat - Secure Access')

        try:
            self.bg_frame = Image.open('images/background1.png')
            photo = ImageTk.PhotoImage(self.bg_frame)
            self.bg_panel = Label(self.window, image=photo)
            self.bg_panel.image = photo
            self.bg_panel.pack(fill='both', expand='yes')
        except:
            self.window.configure(bg='black')

        self.lgn_frame = Frame(self.window, bg='#1a1a1a', width=950, height=600)
        self.lgn_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.lgn_frame.config(highlightbackground="#3047ff", highlightthickness=2)

        self.heading = Label(self.lgn_frame, text="AstraBharat", font=('Arial', 24, "bold"),
                             bg="#1a1a1a", fg='#ffffff')
        self.heading.place(relx=0.5, y=30, anchor=CENTER)

        try:
            self.logo_image = Image.open('images/vector.png').resize((300, 300))
            photo = ImageTk.PhotoImage(self.logo_image)
            self.logo_label = Label(self.lgn_frame, image=photo, bg='#1a1a1a')
            self.logo_label.image = photo
            self.logo_label.place(x=100, y=130)
        except:
            self.logo_label = Label(self.lgn_frame, text="ASTRABHARAT", font=('Arial', 20, "bold"),
                                    bg="#1a1a1a", fg='#3047ff')
            self.logo_label.place(x=120, y=200)

        self.login_panel = Frame(self.lgn_frame, bg='#1a1a1a')
        self.login_panel.place(x=550, y=130, width=300, height=350)
        self.login_panel.config(highlightbackground="#3047ff", highlightthickness=1)

        self.sign_in_label = Label(self.login_panel, text="SECURE ACCESS", bg="#1a1a1a", fg="#3047ff",
                                   font=("Arial", 16, "bold"))
        self.sign_in_label.place(relx=0.5, y=20, anchor=CENTER)

        self.username_label = Label(self.login_panel, text="SECURITY ID", bg="#1a1a1a", fg="#ffffff",
                                    font=("Arial", 12, "bold"))
        self.username_label.place(x=30, y=60)

        self.username_entry = Entry(self.login_panel, bg="#2a2a2a", fg="#ffffff", relief=FLAT,
                                    insertbackground='white', font=("Arial", 12))
        self.username_entry.place(x=30, y=85, width=240, height=35)

        try:
            self.username_icon = Image.open('images/username_icon.png').resize((20, 20))
            photo = ImageTk.PhotoImage(self.username_icon)
            self.username_icon_label = Label(self.login_panel, image=photo, bg='#2a2a2a')
            self.username_icon_label.image = photo
            self.username_icon_label.place(x=5, y=90)
        except:
            pass

        self.password_label = Label(self.login_panel, text="ACCESS CODE", bg="#1a1a1a", fg="#ffffff",
                                    font=("Arial", 12, "bold"))
        self.password_label.place(x=30, y=130)

        self.password_entry = Entry(self.login_panel, bg="#2a2a2a", fg="#ffffff", relief=FLAT,
                                    insertbackground='white', font=("Arial", 12), show="*")
        self.password_entry.place(x=30, y=155, width=240, height=35)

        try:
            self.password_icon = Image.open('images/password_icon.png').resize((20, 20))
            photo = ImageTk.PhotoImage(self.password_icon)
            self.password_icon_label = Label(self.login_panel, image=photo, bg='#2a2a2a')
            self.password_icon_label.image = photo
            self.password_icon_label.place(x=5, y=160)
        except:
            pass

        try:
            self.show_image = ImageTk.PhotoImage(file='images/show.png')
            self.hide_image = ImageTk.PhotoImage(file='images/hide.png')
        except:
            self.show_image = self.hide_image = None

        if self.show_image:
            self.show_button = Button(self.login_panel, image=self.show_image, command=self.show, relief=FLAT,
                                      activebackground="#2a2a2a", borderwidth=0, background="#2a2a2a",
                                      cursor="hand2")
            self.show_button.place(x=240, y=165)

        self.login_button = Button(self.login_panel, text='AUTHENTICATE',
                                   font=("Arial", 12, "bold"),
                                   bg='#3047ff', fg='white',
                                   activebackground='#4057ff', cursor='hand2',
                                   command=self.logon)
        self.login_button.place(x=30, y=220, width=240, height=40)

        self.footer = Label(self.lgn_frame, text="© 2025 AstraBharat Defense Systems. All Rights Reserved.",
                            bg="#1a1a1a", fg="#666666", font=("Arial", 10))
        self.footer.place(relx=0.5, y=570, anchor=CENTER) 

    def show(self):
        self.hide_button = Button(self.login_panel, image=self.hide_image, command=self.hide, relief=FLAT,
                                  activebackground="#2a2a2a", borderwidth=0, background="#2a2a2a",
                                  cursor="hand2")
        self.hide_button.place(x=240, y=165)
        self.password_entry.config(show='')

    def hide(self):
        self.show_button = Button(self.login_panel, image=self.show_image, command=self.show, relief=FLAT,
                                  activebackground="#2a2a2a", borderwidth=0, background="#2a2a2a",
                                  cursor="hand2")
        self.show_button.place(x=240, y=165)
        self.password_entry.config(show='*')

    def logon(self):
        if self.username_entry.get() == "AstraBharat" and self.password_entry.get() == "12345":
            self.window.destroy()
            dashboard()

# ------------------- SPLASH SCREEN -------------------
def splash_screen():
    splash = Tk()
    splash.overrideredirect(True)
    splash.configure(bg='white')
    width, height = 850, 650
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")

    try:
        gif_path = 'images/splash.gif'
        gif = Image.open(gif_path)
        frames = [ImageTk.PhotoImage(frame.copy().convert('RGBA')) for frame in ImageSequence.Iterator(gif)]
        frame_count = len(frames)

        label = Label(splash, bg='white')
        label.pack(expand=True)

        def animate(index=0):
            label.configure(image=frames[index])
            splash.after(80, animate, (index + 1) % frame_count)

        animate()
    except Exception as e:
        print("Splash error:", e)
        Label(splash, text="ASTRABHARAT", font=("Arial", 24), bg="white").pack(expand=True)

    def launch_login():
        splash.destroy()
        window = Tk()
        LoginPage(window)
        window.mainloop()

    splash.after(4000, launch_login)
    splash.mainloop()

# ------------------- RUN APP -------------------
if __name__ == "__main__":
    splash_screen()
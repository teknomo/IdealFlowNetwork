# plugins/mouse_plugin.py
import pyautogui as pg
from pynput import mouse
from pynput import keyboard
import json
import keyboard as keybrd
from typing import List
from plugin import PluginInterface

class MousePlugin(PluginInterface):
    '''
    mouse recording and simulation
    To stop recording: do right click twice and ESC
    To stop simulation: ESC
    '''
    def __init__(self):
        self.events: List[dict] = []
        self.full_path = ""  # Folder and file name of the latest mouse recording
        self.isMouseDown=False
        self.isPrint=True
        pg.FAILSAFE = True # to stop, move the mouse to the upper-left corner of the screen
        
        
    def get_actions(self):
        return {
            'start_recording': self.start_recording,
            'save_recording': self.save_recording,
            'play_recording': self.play_recording,
            'move_to': self.on_move,
            'click': self.on_click,
            'scroll': self.on_scroll,
            'press_key': self.on_press,
            'release_key': self.on_release
        }


    def on_move(self, x, y):
        """Track mouse move events."""
        if self.isPrint: print('Left Mouse moved to {0}'.format((x, y)))
        if self.isMouseDown==True:
            self.events.append("pg.mouseDown(button='left', x={0}, y={1})".format(x, y))
        else:
            self.events.append("pg.moveTo{0}".format((x, y))) 


    def on_click(self, x, y, button, pressed):
        """Track mouse button events (down/up)."""
        if self.isPrint: 
            print('{0} {1} at {2}'.format('Pressed' if pressed else 'Released',
            button, (x, y)))
        
        retVal=True
        if pressed==True:
            self.isMouseDown=True
            if button==mouse.Button.left:                
                self.events.append("pg.mouseDown(button='left', x={0}, y={1})".format(x, y))
            else:                
                self.events.append("pg.mouseDown(button='right', x={0}, y={1})".format(x, y))
        else:
            self.isMouseDown=False
            if button==mouse.Button.left:               
                self.events.append("pg.mouseUp(button='left', x={0}, y={1})".format(x, y))                
            else:
                self.events.append("pg.mouseUp(button='right', x={0}, y={1})".format(x, y))                
                # Stop listener when right button is released
                retVal=False
        return retVal
    
    
    def on_scroll(self, x, y, dx, dy):
        """Track mouse wheel events."""
        self.events.append("pg.scroll{0}".format(dy))
        if self.isPrint: print('Scrolled {0} {1} at {2}'.format('down' if dy < 0 else 'up', (dx,dy), (x, y)))

    def on_press(self,key):
        try:
            print('alphanumeric key {0} pressed'.format(key.char))
            if keybrd.is_pressed('esc'):
                print("stop recording....")
                self._stop_recording()
        except AttributeError:           
            print('special key {0} pressed'.format(key))
            if keybrd.is_pressed('esc'):
                print("stop recording....")
                self._stop_recording()
    
    def on_release(self,key):
        print('{0} released'.format(key))        
        if keybrd.is_pressed('esc'):
                print("stop recording....")
                self._stop_recording()
                return False
        if key == keyboard.Key.esc:
            # Stop listener if Esc is pressed
            return False 


    def start_recording(self, start_key: str = 'tab', **kwargs):
        """Start recording mouse events until stop_key is pressed."""        
        self.events = []
        print(f"Mouse recording will start after you press '{start_key.upper()}'")
        keybrd.wait(start_key)  # Start recording after pressing 'enter'
        print(f"Recording started. Press 'ESC' to stop recording.")

        # Start the listener to capture button events and wheel scrolls
        with keyboard.Listener(
            on_press=self.on_press,                    \
            on_release=self.on_release) as k_listener, \
            mouse.Listener(
                on_move=self.on_move,
                on_click=self.on_click,
                on_scroll=self.on_scroll
                ) as m_listener:
                    k_listener.join()
                    m_listener.join()
        return True


    def save_recording(self, filename: str, **kwargs):
        """Save recorded events to a file with timing data."""
        if not self.events:
            print("No events to save.")
            return False
        try:
            if filename=="":
                print("filename was not provided")
                return False
            self.full_path = self.get_unique_filename(filename)  # in PluginInterface
            with open(self.full_path, 'w') as f:
                f.write(json.dumps(self.events))
            print(f"Saved recording to {self.full_path}")
            return True
        except Exception as e:
            print(f"Failed to save recording: {str(e)}")
            return False

    def play_recording(self, filename: str, speed: str ='Medium', **kwargs):
        """Play back mouse events from a recording file."""
        try:
            self.events = self._load_data(filename)
            if not self.events:
                print("No events found.")
                return
            else:
                print(f"{len(self.events)} events was loaded.")
            
            if speed.lower()=='fast':
                pg.PAUSE = 0.01    # smaller is speedier
            elif speed.lower()=='medium':
                pg.PAUSE = 0.05
            elif speed.lower()=='slow':
                pg.PAUSE = 0.1
            
            print("Simulating Mouse events. Press 'ESC' to interupt.")
            for event in self.events:
                    if self.isPrint: print(event)
                    eval(event)
                    if keybrd.is_pressed('esc'):
                        print("Simulation interrupted.")
                        break
            print("Playback completed.")
            return True
        except Exception as e:
            print(f"Error in play_recording: {str(e)}")
            return False

    def _stop_recording(self, **kwargs):
        """Stop recording mouse events."""
        print("stop recording mouse events")
        mouse.Listener.stop
        
        print(f"Recording stopped. Captured {len(self.events)} events.")
        for event in self.events:
            print(f"{event}")
            
    def _load_data(self, filename):
        events=[]
        if filename == "":
            filename = self.full_path
        try:
            with open(filename, 'r') as f:
                events = json.loads(f.read())
                       
            return events
        except FileNotFoundError:
            print(f"Recording file {filename} not found.")
            return 
        except Exception as e:
            print(f"Exception in _load_data: {str(e)}")
            return False


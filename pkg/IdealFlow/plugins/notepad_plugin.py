# plugins/sample_plugin.py
import time
from plugin import PluginInterface
import subprocess
import pyautogui

class NotepadPlugin(PluginInterface):
    def __init__(self):
        self.saved_file_path = None

    def get_actions(self):
        return {
            'run_notepad': self.run_notepad,
            'type_text_in_notepad': self.type_text,
            'save_notepad': self.save_notepad,
            'save_notepad_as': self.save_notepad_as
        }

    def run_notepad(self, **kwargs):
        subprocess.Popen(['notepad.exe'])
        time.sleep(1)  # Wait for Notepad to open

        # Bring Notepad to the foreground
        windows = pyautogui.getWindowsWithTitle('Untitled - Notepad')
        if windows:
            windows[0].activate()
            time.sleep(0.5)
            print("run_notepad")


    def type_text(self, text, **kwargs):
        time.sleep(0.5)
        pyautogui.typewrite(text, interval=0.05)
        print(f"type_text_in_notepad: {text}")


    def save_notepad(self, **kwargs):
        """Quick save if file has already been saved before, otherwise triggers Save As."""
        if self.saved_file_path:
            pyautogui.hotkey('ctrl', 's')  # Quick save if already saved 
        else:
            # Trigger Save As if this is the first save
            self.save_as(**kwargs)


    def save_notepad_as(self, filename, **kwargs):
        """Save As operation and set saved_file_path."""
        # full_path = self.get_result_path(filename) # in PluginInterface
        full_path = self.get_unique_filename(filename) # in PluginInterface
        
        # Open the Save As dialog and type the file path
        pyautogui.hotkey('ctrl', 'shift', 's')  # Shift+Ctrl+S to open Save As dialog
        time.sleep(0.5)
        pyautogui.typewrite(full_path)
        time.sleep(0.5)
        pyautogui.press('enter')
        time.sleep(0.5)

        # Confirm replace dialog if it appears
        pyautogui.press('enter')
        
        # Update saved_file_path to track that the file has been saved
        self.saved_file_path = full_path
        
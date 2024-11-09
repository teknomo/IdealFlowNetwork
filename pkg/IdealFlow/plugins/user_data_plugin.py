#plugins/user_data_plugin.py
from Automation import PluginInterface
from typing import Callable, Optional

class UserDataPlugin(PluginInterface):
    """
    UserDataPlugin is a plugin for the automation framework that allows 
    users to input, process, and output data through three main actions.

    Actions:
        - get_user_input: Prompts the user to enter data and stores it in `automation.data`.
        - process_data: Processes the data from `automation.data` with an optional modifier.
        - output_data: Outputs the original and processed data from `automation.data`.

    Example Usage:
        ```python
        # In main script
        from Automation import Automation, PluginManager

        # Initialize the automation framework
        af = Automation("User Data Automation")

        # Custom modifier function for process_data
        def custom_modifier(data):
            return data[::-1]  # Example: Reverse the input string

        # Define nodes and actions with parameters
        af.add_node("a", action="get_user_input", params={"automation": af})
        af.add_node("b", action="process_data", params={"automation": af, "modifier": custom_modifier})
        af.add_node("c", action="output_data", params={"automation": af})

        # Define dependencies and execute
        af.assign(["a", "b", "c"])
        af.execute()
        ``` 
    """
    def get_actions(self) -> dict:
        """
        Returns a dictionary of action names mapped to their corresponding methods.

        Returns:
            dict: Dictionary with action names as keys and method references as values.
        """
        return {
            "get_user_input": self.get_user_input,
            "process_data": self.process_data,
            "output_data": self.output_data
        }

    def get_user_input(self, automation: 'Automation') -> None:
        """
        Prompts the user for input and stores the data in the `automation.data` dictionary under the key 'user_input'.

        Parameters:
            automation (Automation): The automation instance where the data will be stored.

        Example:
            >>> plugin = UserDataPlugin()
            >>> plugin.get_user_input(automation)
            Enter some data: Hello
            Data 'Hello' saved to automation.data['user_input'].
        """
        user_input = input("Enter some data: ")
        automation.data["user_input"] = user_input
        print(f"Data '{user_input}' saved to automation.data['user_input'].")

    def process_data(self, automation: 'Automation', modifier: Optional[Callable[[str], str]] = None) -> None:
        """
        Processes the user input from `automation.data` and stores the result in `automation.data` under the key 'processed_data'.
        By default, the data is converted to uppercase, but a custom modifier function can be provided.

        Parameters:
            automation (Automation): The automation instance containing the data.
            modifier (Callable, optional): A custom function to modify the data. Defaults to converting the input to uppercase.

        Example:
            >>> plugin = UserDataPlugin()
            >>> plugin.process_data(automation, modifier=lambda x: x[::-1])  # Reverses the string
            Data processed to 'olleH' and saved to automation.data['processed_data'].
        """
        input_data = automation.data.get("user_input", "")
        processed_data = modifier(input_data) if modifier else input_data.upper()  # Default: Convert to uppercase
        automation.data["processed_data"] = processed_data
        print(f"Data processed to '{processed_data}' and saved to automation.data['processed_data'].")

    def output_data(self, automation: 'Automation') -> None:
        """
        Outputs the user input and processed data stored in `automation.data`.

        Parameters:
            automation (Automation): The automation instance from which the data will be retrieved and printed.

        Example:
            >>> plugin = UserDataPlugin()
            >>> plugin.output_data(automation)
            User Input: Hello
            Processed Data: HELLO
        """
        print("User Input:", automation.data.get("user_input", "No input found"))
        print("Processed Data:", automation.data.get("processed_data", "No processed data found"))

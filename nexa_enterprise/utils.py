import sys
from functools import partial
from nexa_enterprise.constants import (
    EXIT_COMMANDS,
    EXIT_REMINDER,
)

# For prompt input based on the platform
if sys.platform == "win32":
    import msvcrt
else:
    from prompt_toolkit import prompt, HTML
    from prompt_toolkit.styles import Style

    _style = Style.from_dict(
        {
            "prompt": "ansiblue",
        }
    )

    _prompt = partial(prompt, ">>> ", style=_style)

def light_text(placeholder):
    """Apply light text style to the placeholder."""
    if sys.platform == "win32":
        return f"\033[90m{placeholder} (type \"/exit\" to quit)\033[0m"
    else:
        return HTML(f'<style color="#777777">{placeholder} (type "/exit" to quit)</style>')

def strip_ansi(text):
    """Remove ANSI escape codes from a string."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def nexa_prompt(placeholder: str = "Send a message ...") -> str:
    """Display a prompt to the user and handle input."""
    if sys.platform == "win32":
        try:
            hint = light_text(placeholder)
            hint_length = len(strip_ansi(hint))

            # Print the prompt with placeholder
            print(f">>> {hint}", end='', flush=True)

            # Move cursor back to the start of the line
            print('\r', end='', flush=True)
            print(">>> ", end='', flush=True)

            user_input = ""
            while True:
                char = msvcrt.getch().decode()
                if char == '\r':  # Enter key
                    break
                elif char == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                elif char == '\x04':  # Ctrl+D (EOF)
                    raise EOFError
                elif char in ('\x08', '\x7f'):  # Backspace
                    if user_input:
                        user_input = user_input[:-1]
                        print('\b \b', end='', flush=True)
                else:
                    user_input += char
                    print(char, end='', flush=True)

                if len(user_input) == 1:  # Clear hint after first character
                    print('\r' + ' ' * (hint_length + 4), end='', flush=True)
                    print(f'\r>>> {user_input}', end='', flush=True)

            print()  # New line after Enter

            if user_input.lower().strip() in EXIT_COMMANDS:
                print("Exiting...")
                sys.exit(0)
            return user_input.strip()
        except KeyboardInterrupt:
            print(EXIT_REMINDER)
            return
        except EOFError:
            print("Exiting...")
            sys.exit(0)
    else:
        try:
            user_input = _prompt(placeholder=light_text(placeholder)).strip()

            # Clear the placeholder if the user pressed Enter without typing
            if user_input == placeholder:
                user_input = ""

            if user_input.lower() in EXIT_COMMANDS:
                print("Exiting...")
                sys.exit(0)
            return user_input
        except KeyboardInterrupt:
            print(EXIT_REMINDER)
            return
        except EOFError:
            print("Exiting...")

    sys.exit(0)
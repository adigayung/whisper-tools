from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)  # untuk reset otomatis warna terminal

def log_print(message, level="INFO", debug=True):
    if not debug:
        return

    if level == "INFO":
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {message}")
    elif level == "STEP":
        print(f"{Fore.BLUE}{Style.BRIGHT}➤ {message}{Style.RESET_ALL}")
    elif level == "SUCCESS":
        print(f"{Fore.GREEN}{Style.BRIGHT}✔ {message}{Style.RESET_ALL}")
    elif level == "SUCCESS2":
        print(f"{Fore.CYAN}{Style.BRIGHT}✔ {message}{Style.RESET_ALL}")
    elif level == "WARNING":
        print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}")
    elif level == "ERROR":
        print(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}")
    else:
        print(message)

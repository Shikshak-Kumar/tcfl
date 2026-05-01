import datetime

# ANSI Colors for Professional Terminal Output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    @staticmethod
    def _timestamp():
        return datetime.datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def info(msg, prefix="INFO"):
        print(f"[{Logger._timestamp()}] {Colors.BLUE}ℹ {prefix}:{Colors.ENDC} {msg}")

    @staticmethod
    def success(msg, prefix="SUCCESS"):
        print(f"[{Logger._timestamp()}] {Colors.GREEN}✔ {prefix}:{Colors.ENDC} {msg}")

    @staticmethod
    def warning(msg, prefix="WARNING"):
        print(f"[{Logger._timestamp()}] {Colors.WARNING}⚠ {prefix}:{Colors.ENDC} {msg}")

    @staticmethod
    def error(msg, prefix="ERROR"):
        print(f"[{Logger._timestamp()}] {Colors.FAIL}✖ {prefix}:{Colors.ENDC} {msg}")

    @staticmethod
    def header(msg):
        line = "=" * (len(msg) + 4)
        print(f"\n{Colors.HEADER}{Colors.BOLD}{line}")
        print(f"  {msg}")
        print(f"{line}{Colors.ENDC}\n")

    @staticmethod
    def section(msg):
        print(f"\n{Colors.CYAN}{Colors.BOLD}--- {msg} ---{Colors.ENDC}")

    @staticmethod
    def table(headers, rows):
        # Basic terminal table formatter
        if not rows: return
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))
        
        # Format string
        fmt = " | ".join([f"{{:<{w}}}" for w in widths])
        separator = "-+-".join(["-" * w for w in widths])
        
        print(f"{Colors.BOLD}{fmt.format(*headers)}{Colors.ENDC}")
        print(separator)
        for row in rows:
            print(fmt.format(*row))
        print("")

logger = Logger()

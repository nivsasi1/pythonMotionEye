import json
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import psutil
import inputToSignUp
import socket

with open("login.html", "r") as index_file2:
    login_page = index_file2.read()

with open("index.html", "r") as index_file:
    buttons_page = index_file.read()

running_eye = False
running_hand = False
flag = False


def turn_on(algoName):
    algo_thread = threading.Thread(target=run_algorithm, args=(algoName,))
    algo_thread.start()


def run_algorithm(algoName):
    os.system(algoName)


def turn_off(algoName):
    script_name = algoName
    # I noticed two of the same name, closing the one who isnt script will close the program
    not_the_right_one = 'venv\\Scripts'
    # Running over every process untill finding the deired one
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the process is a Python process
            if proc.info['name'] == 'python.exe':
                # if so, check if it is the desired one
                cmdline_args = proc.cmdline()
                for a in cmdline_args:
                    if not not_the_right_one in a:
                        if script_name in a:
                            print("Running algo found!")
                            print("PID:", proc.info['pid'])
                            print("Command:", ' '.join(cmdline_args))
                            # Closing
                            proc.terminate()
                            break  # Exit the loop after finding the first matching process
                    else:
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Handling any exceptions that may occur while accessing process information


def get_ip_address():
    # create a temporary socket to get the IP address
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(('8.8.8.8', 80))  # connect to a public DNS server
    address = sock.getsockname()[0]  # get the socket's IP address
    sock.close()
    print(address)


class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global flag
        if self.path == '/your_login_endpoint':
            user = False
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            username = data['username']
            password = data['password']

            encrypted_data = inputToSignUp.encrypt_decrypt(username)
            encrypted_data2 = inputToSignUp.encrypt_decrypt(password)
            data2 = {"username": encrypted_data, "password": encrypted_data2}

            json_file_path = "mamaMia.json"
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
            for user_data in data:
                if user_data == data2:
                    user = True
                    break
                else:
                    user = False

            if user:
                flag = True
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'message': 'Login successful'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                flag = False
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'message': 'Invalid username or password'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            flag = False
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        global flag
        global running_eye, running_hand
        parsed = urlparse(self.path)
        path = parsed.path.lower()
        print(path)
        if path == "/":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(login_page
                             .encode())

        elif path == "/index.html" and flag:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(buttons_page
                             .encode())

        elif path.startswith("/turn_eye"):
            if path == "/turn_eye_on":
                turn_on('python eyeAlgo.py')
            else:
                turn_off('eyeAlgo.py')

            self.send_response(200)
            self.end_headers()
            self.wfile.write(buttons_page
                             .encode())

        elif path.startswith("/turn_hand"):
            if path == "/turn_hand_on":
                turn_on('python handFile.py')
            else:
                turn_off('handFile.py')

            self.send_response(200)
            self.end_headers()
            self.wfile.write(buttons_page
                             .encode())

        else:
            self.send_error(404)
            self.end_headers()


httpd = HTTPServer(('localhost', 80), MyHandler)
get_ip_address()
print("Running...")
httpd.serve_forever()

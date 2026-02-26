from interview_manager import start_interview

while True:
    start_interview()
    choice = input("\nStart Again? yes/no: ")
    if choice.lower() != "yes":
        break
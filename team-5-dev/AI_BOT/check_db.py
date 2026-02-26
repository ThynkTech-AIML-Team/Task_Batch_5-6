from database_module import fetch_responses

data = fetch_responses()

if not data:
    print("No records found in database.")
else:
    print("\nSaved Interview Records:\n")
    for row in data:
        print(row)
import json

def encrypt_decrypt(inpString):
    # Any character value will work for XOR key
    xorKey = 'P'

    # calculate length of input string
    length = len(inpString)

    # perform XOR operation of key
    # with every character in string
    for i in range(length):
        inpString = (inpString[:i] +
                     chr(ord(inpString[i]) ^ ord(xorKey)) +
                     inpString[i + 1:])
    return inpString

#json = mostly used to move data with from servers and web application (JS)
#or to save the data, file that allow me to save information as a list with 2 names for value, like dictionary

if __name__ == '__main__':

    string_to_encrypt = "nivsasi1"
    string_to_encrypt2 = "nivos1110"

    string_to_encrypt3 = "ofirTheKing1"
    string_to_encrypt4 = "ofiros1110"

    encrypted_data = encrypt_decrypt(string_to_encrypt)
    encrypted_data2 = encrypt_decrypt(string_to_encrypt2)
    encrypted_data3 = encrypt_decrypt(string_to_encrypt3)
    encrypted_data4 = encrypt_decrypt(string_to_encrypt4)



    # create a dictionary containing the encrypted data

    data = {"username": encrypted_data, "password": encrypted_data2}
    data2 = {"username": encrypted_data3, "password": encrypted_data4}
    bigdata = [data, data2]

    # save the dictionary to a JSON file
    json_file_path = "mamaMia.json"
    with open(json_file_path, "w") as json_file:
        json.dump(bigdata, json_file)
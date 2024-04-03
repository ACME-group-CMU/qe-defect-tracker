
import os
from dotenv import load_dotenv

class Utility(object):

    def __init__(self,debug_obj=None):

        self.debug_obj = debug_obj

    def checkEnvironmentalVars(self,env_var):

        load_dotenv()
        env_var_key = ""
        try:
            env_var_key = os.environ[env_var]
            return env_var_key
        except KeyError:
            message = f"The environment variable {env_var} not set. Set this before running this program"
            self.debug_obj.debug_log(message,type="Error")
            return False

    #filename = name of filepath/filename
    def checkIfFileExists(self,filename):
        file_found = False

        item_list = os.listdir()
        for item in item_list:
            if filename in item:
                file_found = True

        return file_found

        #filename = name of filepath/filename
    def createNewDirectory(self,dir_path):
        result = False
        try:
            os.mkdir(dir_path)
            result = True
        except:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Failed to create path: {dir_path}")
        return result

    #filename = name of filepath/filename
    def changeDirectory(self,dir_path):
        result = False
        try:
            os.chdir(dir_path)
            result = True
        except:
            self.debug_obj.debug_log(f"Current working directory: {os.getcwd()}")
            self.debug_obj.debug_log(f"Unable to change location to {dir_path}",type="WARNING")
        return result

    #If precise is False, if term_to_delete is in the file, the file is deleted
    #If precise is True, only the exact name will result in a deletion
    def deleteFiles(self,term_to_delete,precise = True):

        self.debug_obj.debug_log(f"Attempting to delete files.")
        try:
            item_list = os.listdir()
            for item in item_list:

                if precise == True:
                    if term_to_delete == item:
                        os.remove(item)
                        self.debug_obj.debug_log(f"Deleted file: {item}")

                if precise == False:
                    if term_to_delete in item:
                        os.remove(item)
                        self.debug_obj.debug_log(f"Deleted file: {item}")

        except:
            self.debug_obj.debug_log(f"Unable to delete any files.",type="WARNING")



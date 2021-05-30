import os
import logging
from typing import Tuple


def getFiles(dirName, allowedExtensions : Tuple = None):
    """Gets all files in the directory recursively

    Args:
        dirName (str): directory to search for files

    Returns:
        list: list of paths of files
    """
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getFiles(fullPath, allowedExtensions= allowedExtensions)
        else:
            if allowedExtensions:
                if entry.endswith(allowedExtensions):
                    allFiles.append(fullPath)
            else:
                allFiles.append(fullPath)
                
    return allFiles

def createDir(dirName):
    """Creates the directory if not already exists

    Args:
        dirName (str): directory path
    """
    if not os.path.isdir(dirName):
        os.makedirs(dirName)
        logging.info(f"Created output directory {dirName}")

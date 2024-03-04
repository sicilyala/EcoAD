import os 
import sys


class AutoLog:
    """
    save log automatically
    """

    def __init__(self, filepath, filename, if_note=False, stream=sys.stdout):
        self.terminal = stream
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # create a note.txt
        if if_note:
            note_name = filepath + 'note.txt'
            if not os.path.exists(note_name):
                file = open(note_name, 'w')
                file.write('-----Configuration note-----' + '\n')
                file.close()

        self.log = open(filepath + "/" + filename, 'a')  # 文件末尾追加写入

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
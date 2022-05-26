from PyQt5.QtWidgets import QFrame


class Statusbar_VLine(QFrame):
    def __init__(self):
        super(Statusbar_VLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
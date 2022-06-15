from PyQt5.Qt import QPalette
from PyQt5.QtGui import QColor


class ThemePalette(QPalette):
    def __init__(self):
        super(QPalette, self).__init__()


        windowText_color = QColor(252, 252, 252)
        button_color = QColor(49, 54, 59)
        light_color = QColor(64, 70, 76)
        midlight_color = QColor(54, 59, 64)
        dark_color = QColor(25, 27, 29)
        mid_color = QColor(37, 41, 44)
        text_color = QColor(252, 252, 252)
        brightText_color = QColor(255, 255, 255)
        buttonText_color = QColor(252, 252, 252)
        base_color = QColor(27, 30 ,32)
        window_color = QColor(42, 46, 50)
        shadow_color = QColor(18, 20, 21)
        highlight_color = QColor(61, 174, 233)
        highlightedText_color = QColor(252, 252, 252)
        link_color = QColor(29, 153, 243)
        linkVisited_color = QColor(155, 89, 182)
        alternateBase_color = QColor(35, 38, 41)
        toolTipBase_color = QColor(49, 54, 59)
        toolTipText_color = QColor(252, 252, 252)
        placeholderText_color = QColor(252, 252, 252, 128)
        
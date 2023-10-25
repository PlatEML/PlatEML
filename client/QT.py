import os
import sys

from PyQt5 import QtCore, QtWidgets, uic

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    path_ui = os.path.join(os.path.dirname(__file__), "page.ui")
    window = uic.loadUi(path_ui)
    window.setWindowTitle("Evolutionary Machine Learning")
    window.widget.load(QtCore.QUrl("http://localhost:8050//"))
    window.show()
    sys.exit(app.exec_())

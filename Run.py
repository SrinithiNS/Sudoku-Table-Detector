'''Run this file to run the application'''
from MainUI import MainUI
import os
modeltype = "CNN"

if os.path.exists("cnn.hdf5"):
    pass
else:
    print("cnn.hdf5 not found...")
    print("Train and save model by running newmodel.py")
    exit(0)
MainUIobj = MainUI(modeltype)
MainUIobj.mainloop()
MainUIobj.cleanup()

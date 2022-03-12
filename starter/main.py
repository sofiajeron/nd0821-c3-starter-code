# Put the code for your API here.
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/libd/dvc")


os.system("dvc remote add -df s3-bucket s3://udacitycoursebucket")



dvc run -n clean_data \
          -d ../starter/ml/clean.py -d census.csv \
          -o clean_data.csv \
          python ../starter/clean_data.py
The file structure: 

Folder
- /api_use_ref/: The API that is related to this project, for reference
- /dataset/: Contains the left and right image that is used in Part 2
- /test_result/: Contains the depth map image generated in Part 2
- /source/: Contains the left and right image used in Part 3
- /result/: Contains the depth map image generated in Part 3


To run the code:
- Part 2: python main.py
- Part 3: python zncc_faster_numba.py


package installation
`pip install numpy opencv-contrib-python matplotlib numba`

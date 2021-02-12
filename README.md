# Class202 
Contain the Projects for CM202.


## Project1: WaveFunctionCollaps 

### Description:

This project is a python version game to generate map using the tile in ./Images folder

Each of the image in the folder and their rotation of 90 degree, 180 degree and 270 degree will be shown in the start page
![Screen Shot 2021-01-25 at 6 51 37 PM](https://user-images.githubusercontent.com/24282146/105781803-3284a980-5f41-11eb-83b5-74014eb2e576.png)

Users can select which images they want
![Screen Shot 2021-01-25 at 6 51 14 PM](https://user-images.githubusercontent.com/24282146/105781814-387a8a80-5f41-11eb-81d6-93fb2f0db40d.png)

Then users can define their constraints by selecting the available adjacent image on particalar directions.
![Screen Shot 2021-01-25 at 6 54 48 PM](https://user-images.githubusercontent.com/24282146/105781822-3adce480-5f41-11eb-9a05-3933601f169a.png)

After constructing constraints, the maps are generated using WaveFunctionCollaps algorithm
![Screen Shot 2021-01-25 at 7 08 37 PM](https://user-images.githubusercontent.com/24282146/105781824-3ca6a800-5f41-11eb-9ce7-46d78b826f4c.png)

Users can select the maps to save in the ./ResultMaps/ folder or regenerate the maps.
![Screen Shot 2021-01-25 at 7 08 47 PM](https://user-images.githubusercontent.com/24282146/105781829-3dd7d500-5f41-11eb-969c-36c66702faf4.png)

### How to use:

1. Use __*git clone https://github.com/ZhaoqingT/Class202.git*__
2. Use __*cd Class202*__ , __*cd Project1*__ to go into the Class202/Project1/ directory,
3. Use __*conda env create -f environment.yml*__ to create the environment
4. Use __*source activate CM202*__ to enter the environment (Or *conda activate CM202*)
5. Use __*python3 -m pip install -U pygame --user*__ to install pygame
6. Use __*python3 app.py*__ to run the application

### Alert!

1. WaveFunctionCollaps does not guarantee to generate a result. The image generated is the last iteration if there is no tile to find next.
2. I know constructing constraints might be exhausting but it might be really hard to generate result if you choose A on the bottom of B but forget to choose B on  the top of A.
3. Will have more features in the future. Like
   - ML part to filter the map to generate more prefered map based on previous selections
   - Auto selection part so that users can have less works constructing the constraints.
   - Improve the algorithm to make it generate valid results more probably.
   
   


## Project2: Neural Network 

### Description:
This is a text generator using https://github.com/minimaxir/textgenrnn. I have trained the model with tweet data so that the model is able to generate tweet-style text. The data used to train the model is from https://dataverse.harvard.edu/dataset.xhtml?id=3047332, which is top 20 most followed user on twitter social platform. And I only used the content column to train the model. 


### How to use:
1. Use __*git clone https://github.com/ZhaoqingT/Class202.git*__
2. Use __*cd Class202*__ , __*cd Project2*__ to go into the Class202/Project1/ directory,
3. Use __*pip install textgenrnn*__ to install textgenrnn
4. Use __*python TweetGen.py*__ to generate tweet-style text
5. Use __*python NormalTextGen.py*__ to generate normal text

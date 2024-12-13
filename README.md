# ppp_tetris_ai

# Simple tetris algorithem with 4 weights 

This implementation of a Tetris Player uses the genetic algorithem to optimize a number of weights, which evaluate the optimal position for any given piece. Players are evaluated based on the average number of lines they cleared in a given number of playthroughs. 

Screenshot during training: 

<img width="906" alt="Untitled" src="https://github.com/user-attachments/assets/aba68bbe-510d-4719-8d65-9be0a0c160d7" />

Passable Weights: 

Line clear reward, Aggregate Height, Bumbiness, Holes 

- [0.6463964408855271, 0.34259941535955357, 4.176671512193586, 1.0235185039686558] cleared 500 lines almost optimally and performed in other variations above average
- [2.600608885994828, 1.3607727778207876, 0.7311918425046104, 0.3510132056841997] 955 Lines
- [2.484527990484525, 3.7572371538123885, 3.606698363559004, 1.041341844599832] 640 LInes
- [0.6028086938596067, 3.4133561981164835, 3.7534200486945526, 0.9857318593113606] 786 Lines
- [4.087625452471151, 4.228799786347267, 1.9737959160186491, 0.7074388479183652] 600 Lines
- [0.5589995798669718, 0.21173194413336374, 4.0934409738329665, 0.9692410464877592]


After an inital burst the fitness always maxes out near 220 and stays there. 

Currently there is no consideration of the next piece/s and other weights could be experimented with in the future, such as deepest well as they can only be cleared with line pieces without leaving holes, number of seperate holes in a row, xxx....xxx is better then x.xx.x.x.x even though both have the same number of holes. 

The playback of a player can be initiated with the command: 
```python3 tetris_ai.py --visualize --weights_file best_agent.json --phase 4```

# Advanced tetris algorithem with 6 weights and more improvements 

- This adds adaptive mutation rates which will increase the mutation rate gradually if the ai remains stagnant for a specified number of generation. 
- Adds gradiant based exploration for elite specimen which will try to explore further, indipendent from the score to get out of local minima. 
- Adds more complex mutation to increase genetic diversity. 
- Adds a genetic diversity score which needs to stay above a specific number or the mutation rate will also increase. This makes sure that the population doesn't diverge into one set of genes. 
- Adds tournament based selection for reproduction which takes 3 agents and only the best advances.
- Adds two new weights, heighest stack and 4 deep wells

The advanced tetris algorithem managed to surpass the stagnation point of the simple tetris algorithem, which was an average of 230 line clears and managed to reach an average of 412 line clears after 64 generations. (Taking around 12 hours) 
During Training the highest fitness evaluation was 920 over the 10 attempts. 

# Observations 

Over multible iterations and versions the ai seamed to optimize the agreggate height weight towards ```0.0```, this happened on both the 4 and 6 weight versions. This makes sense as the performance of a single piecd is indipendent of the total stack height most of the time and would only change it by little if it wasn't. 

At the same time the heighest and most important parameters seamed to be the bumbiness and holes, these factors directly influence how the next piece performs and are a significant risk factor when playing. Interestingly the line clear reward was a lot more flexible, hovering between ```0.5``` and ```4.5```

# Game implementation 

<img width="594" alt="Untitled" src="https://github.com/user-attachments/assets/ebf8b024-c9b7-4e82-8dd5-e0cb570f981e" /> 

The algorithem can play the game in a more fvisual style using the pygame library and a command like this: 

python playtetris.py --play --weights 0.6743113139572464 0.0 4.664487829366196 0.9326869578657685 0.0 1.2950522791394365

An implementation into online platforms like tetr.io is easily possible with the new play mode which returns a string of actions for any board state and piece combination. The only problem is that these websites have slightly differant strting positions creating a need to translate the spawning pieces. 

<img width="234" alt="Screenshot 2024-12-12 at 17 03 14" src="https://github.com/user-attachments/assets/5b19470a-4467-4094-ad57-b91d8cea7d0f" /> <img width="230" alt="Untitled" src="https://github.com/user-attachments/assets/c31b0bd2-4c41-4f12-a92c-6a3f9e3bcd9e" /> <img  alt="Untitled" src="https://github.com/user-attachments/assets/1d555bce-33ba-49af-987b-35bf24ec5198" width="315" height="300" />



To get the current piece we make a screenshot of the area where the next piece is and use edge detection to isolate it from the background. Finally we compare it to our library of reference images and see if we find a match. This is 100 reliable due to the very clear distinction between the images. 



Inspiration https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

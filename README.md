# ppp_tetris_ai

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


Inspiration https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

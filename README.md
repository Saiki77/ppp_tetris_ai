

# ppp_tetris_ai

## Simple Tetris Algorithm with 4 Weights

This implementation of a Tetris Player uses the genetic algorithm to optimize a number of weights, which evaluate the optimal position for any given piece. Players are evaluated based on the average number of lines they cleared in a given number of playthroughs. 

Screenshot during training: 

<img width="906" alt="Untitled" src="https://github.com/user-attachments/assets/aba68bbe-510d-4719-8d65-9be0a0c160d7" />

### Passable Weights: 

- **Line clear reward**, **Aggregate Height**, **Bumpiness**, **Holes**
  
  Example weights:
  - [0.6463964408855271, 0.34259941535955357, 4.176671512193586, 1.0235185039686558]: Cleared 500 lines almost optimally and performed above average in other variations.
  - [2.600608885994828, 1.3607727778207876, 0.7311918425046104, 0.3510132056841997]: 955 Lines
  - [2.484527990484525, 3.7572371538123885, 3.606698363559004, 1.041341844599832]: 640 Lines
  - [0.6028086938596067, 3.4133561981164835, 3.7534200486945526, 0.9857318593113606]: 786 Lines
  - [4.087625452471151, 4.228799786347267, 1.9737959160186491, 0.7074388479183652]: 600 Lines
  - [0.5589995798669718, 0.21173194413336374, 4.0934409738329665, 0.9692410464877592]: Various performances.

After an initial burst, the fitness always maxes out near 220 and stays there. 

Currently, there is no consideration of the next piece(s), and other weights could be experimented with in the future, such as:
- Deepest well (useful for line pieces without leaving holes).
- Number of separate holes in a row (e.g., `xxx....xxx` is better than `x.xx.x.x.x` even though both have the same number of holes).

The playback of a player can be initiated with the command: 
```python
python3 tetris_ai.py --visualize --weights_file best_agent.json --phase 4
```

## Advanced Tetris Algorithm with 6 Weights and More Improvements

- Adds adaptive mutation rates, which increase gradually if the AI remains stagnant for a specified number of generations.
- Includes gradient-based exploration for elite specimens, enabling them to explore further, independently of the score, to escape local minima.
- Implements more complex mutations to increase genetic diversity.
- Tracks a genetic diversity score, ensuring the population doesn't converge on one set of genes. If diversity drops too low, the mutation rate increases.
- Introduces tournament-based selection for reproduction, selecting the best of three agents.
- Adds two new weights: **highest stack** and **4-deep wells**.

The advanced algorithm surpassed the stagnation point of the simple algorithm (average of 230 line clears) and achieved an average of 412 line clears after 64 generations (approximately 12 hours). During training, the highest fitness evaluation was 920 over 10 attempts.

## Observations

Over multiple iterations and versions, the AI seemed to optimize the **aggregate height** weight toward `0.0`. This occurred in both the 4-weight and 6-weight versions. This makes sense because the performance of a single piece is mostly independent of the total stack height, with only minor effects.

At the same time, the **bumpiness** and **holes** weights were the most critical parameters. These factors directly influence how the next piece performs and are significant risk factors. Interestingly, the **line clear reward** weight was more flexible, hovering between `0.5` and `4.5`.

## Game Implementation

<img width="594" alt="Untitled" src="https://github.com/user-attachments/assets/ebf8b024-c9b7-4e82-8dd5-e0cb570f981e" /> 

The algorithm can play the game in a more visual style using the `pygame` library. Example command: 

```python
python playtetris.py --play --weights 0.6743113139572464 0.0 4.664487829366196 0.9326869578657685 0.0 1.2950522791394365
```

Integration into online platforms like `tetr.io` is possible with the new play mode, which returns a string of actions for any board state and piece combination. However, these websites have slightly different starting positions, requiring translations for spawning pieces.

<img width="234" alt="Screenshot 2024-12-12 at 17 03 14" src="https://github.com/user-attachments/assets/5b19470a-4467-4094-ad57-b91d8cea7d0f" /> <img width="230" alt="Untitled" src="https://github.com/user-attachments/assets/c31b0bd2-4c41-4f12-a92c-6a3f9e3bcd9e" /> <img  alt="Untitled" src="https://github.com/user-attachments/assets/1d555bce-33ba-49af-987b-35bf24ec5198" width="315" height="300" />

### Piece Detection

To get the current piece, the algorithm:
1. Takes a screenshot of the area where the next piece is displayed.
2. Uses edge detection to isolate it from the background.
3. Compares it to a library of reference images to find a match. This method is 100% reliable due to the clear distinction between the images.

---

**Inspiration**: [Tetris AI: The Near-Perfect Player](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/) 

--- 


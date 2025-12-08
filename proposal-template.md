# CST463 Project Proposal
## Team members

- Logan Druley
- Alberto Garcia
- Roberto Palacios

## Project idea

Through the use of Elo chess rating systems, we can quantify a player's skill in relation to those they play against. However, this requires several games and multiple participants to produce a valid assessment of a player's skill. Our aim is to produce a machine learning model that can predict a player's Elo rating by analyzing sequences of board positions as FEN strings. With this model in place, we can analyze player behavior to better match them with players of similar skill or dynamically adjust AI opponent difficulty.

Our baseline for the Elo prediction model will lie in the error range of 100–200 Elo points, as this range represents an opponent with a moderate but not unsurmountable skill gap. Our stretch goals include an analysis of which moves and board states our model considers to be indicative of skill level. Additionally, we want to use our model to segment Elo scores into difficulty ranges that can be further used to tune chess AI difficulty.

##  Data set

For this project, we plan on using the Lichess database of rated chess games in PGN format. This represents a collection of trillions of games with move sequences, rating information, and game metadata. From this, we will train a number of smaller trial models before setting out on training a more sizable model.

## Architectures to utilize

As we want to see the importance that each move has on the overall game, we want to focus on an architecture capable of self-attention. To this end, we plan on adopting a transformer encoder architecture, as this will help us connect decisions made across the game and see how they affect the end state. Given the nature of the FEN strings we will be working with, we can leverage the transformer’s ability to process sequential information in parallel for efficiency’s sake. However, we may first build an LSTM for this same purpose so that we can later compare the attention-driven approach against this more traditional network.

## Techniques to leverage

Self-attention is key to this project, as we expect individual board states or sequences of such states to act as strong indicators of player skill. Beyond this, we plan to use the FEN string to help us encode positions for each of our tokens, as this will also play into how our model interprets the board states.

## Expected computational requirements

To train our models, we have access to Oregon’s Regional Computing Accelerator (ORCA), which provides us with 192 GB of VRAM, 64 CPU cores, and 576 GB of RAM. If this is not enough compute for us, then frankly we’re probably doing something wrong on our end. We can scale down our model if needed, but we’ve processed similar amounts of data on this cluster and don’t expect too many issues.
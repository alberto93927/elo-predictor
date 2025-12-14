# CST463 Project Update

## Team members

- Alberto Garcia
- Roberto Palacios
- Logan Druley

## Updated Project Goal

Our goal still remains predicting Elo rankings based on player board states in a game. However, we are likely scaling down our efforts as training even on modest datasets takes longer than we have the time or computing power for. Given that even our intial model perform quite well for prediciting Elo ratings, our next major focus with be quantifying the importance of moves within a game using attention.

## Baseline Objective

Our baseline objecttive was a to train a models that can predict, withing 100-250 Elo points, the Elo ranking of a player based on their performance in a game. Our intitial models produced model whos test accuracy lay in this range. As such, our baseline objective is complete but needs furhter fleshing out to confirm that we can consistently achieve this performance or even improve our performance.

## Status on baseline objective

We plan on train more complex model in order to guage how much room for improvement there truly is. Furthermore, as we are interested in the impact individual moves have course of a game, we plan on focusing our efforts on attention models that can help us visulaize this importance.

## Plan for next week

This week, we plan to optimize our current models and the expand the amount of data we are training on. This will allow us to compare the performance of larger models against their "quick test" counterparts. Furthermore, we plan on introduing self-attention to our transformer. This is partially for our own edification but also so that we can see the relationships between board states across whole games.

## Contributions

In their own words, what are the contributions of each team member?

### Alberto Garcia

I initialized the repository and was involved in early architectural decision making. I debugged and fixed the data preprocessing pipeline. There were issues with the initial PGN parser, where games weren't being parsed due to missing newlines when joining lines and overly broad move detection that matched dates instead of actual chess moves. I then resolved shape mismatch errors in both the Transformer and LSTM models where pooling output wasn't being properly flattened before the projection layer, and fixed datatype compatibility issues. I created a quick pipeline test script to verify both models train correctly on small datasets, and assessed computational feasibility across different hardware configurations. I updated the documentation with clear setup and testing instructions for cross-platform compatibility.

### Roberto Palacios

I optimized and trained an LSTM model to act as our baseline for this project. I also began work on the notebook that will become part of our final project submission. This largely consisted of organizing Alberto's work into functions that were readily available to be used through the notebook. I then ran a few tests of the LSTM model before settling on one that used a random sample of our available data for training, validation and testing. This is the model that acts as our initial baseline.

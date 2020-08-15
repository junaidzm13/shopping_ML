# shopping_ML

### Designed, developed and implemented an AI to predict whether online shopping customers will complete a purchase.

## How to run?
  - In the repository, run **python shopping.py shopping.csv**

## Background
When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don't end up going through with a purchase during that web browsing session. It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer if the website believes the user isn't planning to complete the purchase. How could a website determine a user's purchasing intent? That’s where our machine learning program will come in.

I have implemented a nearest-neighbor classifier to solve this problem. Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser they’re using, etc. — my classifier predicts whether or not the user will make a purchase. Obviously it won’t be perfectly accurate but it is much better than guessing randomly. To train my classifier, I have used data from a shopping website from about 12,000 users sessions.

## Acknowledgements
  - Data set provided by [Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)](https://link.springer.com/article/10.1007/s00521-018-3523-0)
  - Special thanks to CS50AI 2020: Intriduction to Artificial Intelligence with Python course's managment, lecturers and tutors.

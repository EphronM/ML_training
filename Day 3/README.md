
# Day 3 of ML Training

# Know more about samplings 

    1. What is Sampling?
    2. What is a Sample against a Population?
    3. What are the assumptions we make on the sample around its ability to
       represent the characteristics of the population?
    4. The ideas around sampling with and without replacement.
    5. When a sample can go haywire?
    6. How do we estimate the discrete distribution of a sample and compare it 
       against the population distribution?
    7. How do we estimate the quality of the sample by checking the deviation in distributions?
    8. Why are duplicates expected in sampling with replacement?
    9. A simple experiment to check if the unique data points in the sample with replacement
       are around 63% of the sample size.
    10. What is random sampling?
    11. What is the notion of probabilistic sampling?
    12. What are PMF and PDF? How to compute PMF from a frequency distribution?
    13. How do we compute the cumulative mass function from a discrete distribution?
    14. The idea behind the implementation of a Random sampler that follows
        a data distribution.
    15. Visualizing and understanding a continuous distribution.
    16. Experimenting with the drawing of random samples following a distribution.



## Assignment

- Implement the random sampler that accepts a PMF and the # of required data points as inputs.
- Generate random numbers in the range [1 to K] where K is in {1k, 10k, 100k, 1m, 10m}, which is 
  equivalent to sampling with replacement. Check if the proportion of unique numbers in the sample is ~63%.
- Create an arbitrary PMF for X={x|0<=x<=9}. Now draw random samples of size K in {1k, 10k, 100k, 1m, 10m} 
  following that PMF. Verify if the estimated distribution of the sample is close to the PMF that you originally constructed.

Happy Coding!
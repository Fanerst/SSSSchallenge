# SSSSchallenge

Codes for coding challenge@Song-Shan-Hu Sping School. All results can be found on `ans1.txt` and `ans2.txt`

## Question 1

There are three approaches to solve this question.

The first approach is by sampling all vertices of FVS and summing up others to calculate $\log Z$ exactly, try  

```python
cd src/; python exact.py
```

to get results and running time.

The second approach is contracting tensor networks, run

```python
python src/TNchallenge.py
```

the result is identical to the first one, but the running time will be highly reduced.

The third approach is by running Variational Autoregressive Networks(VAN) on the antiferromagnetic buckyball. Since VAN is basically a vairational method, thus the exact result will be hard to obtain. We put it here just to show that VAN is suitable to solve questions like calculating $\log Z$ (free energy) or quantities like that. Codes of VAN can be founded at `src/VAN.py`.

## Question 2

From the begining, we used VAN to find ground states(there are tons of methods to do the same task, like simulated annealing), collected over 15000 of them, and then analyzed their patterns.

After brainstorming, we found two ways to solve it. The story is long, maybe we will update it later.

The results can both be calculated by

```python
cd src/; python findfactor.py
```

and

```C++
cd src/; g++ -o gs1 gs.cpp;  ./gs1
```

## Contributors

Sujie Li, Pengfei Zhou,  Feng Pan. All come from ITP, CAS, directed by Prof. Pan Zhang

## Acknowledgments

We want to thank Yifan Qu of many helpful insights and all the staff and students of this spring school. We really enjoy this wonderful week.
Implementation of Point-Based Value Iteration
=============================================

Point-based value iteration (PBVI) is an approximate method for solving
partially observable Markov decision processes (POMDPs). I'm implementing the
algorithm from:

<a href="#pbvi-article"></a>
[Joelle Pineau, Geoffrey Gordon, Sebastian Thrun. Point-based value iteration:
An anytime algorithm for POMDPs. In IJCAI,
2003.](http://ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf)


Done
----

- Implement [PBVI](pbvi.py).
- Implement a [naive version](naive_pbvi.py) that is slower, but easier to read.
- [Test](pomdp_play.py) them on a two-state problem from AIMA. Compare value
  function with AIMA. – Looks good.
- [Test](pomdp_test.py) it on the [Tiger95](http://www.pomdp.org/examples/)
  POMDP. Compare value function with the result of
  [pomdp-solve](http://www.pomdp.org/code/index.html). – Looks okay. pomdp-solve
  returns a value function with more segments, but the policy is the same, I
  think.
- Make a [tool](https://github.com/rmoehn/pomdp2json) to convert from Anthony
  Cassandra's POMDP file format to JSON.


To do
-----

- Maybe find out why my implementation returns a value function with fewer
  segments than pomdp-solve on Tiger95.
- Make an OpenAI Gym environment that can run most POMDPs specified in Anthony
  Cassandra's POMDP file format converted to JSON. ‘Most’ means that the reward
  function will be limited to depend on the action and the state before taking
  the action.
- Maybe make it easier to obtain converted POMDP files.
- Run my implementation on the Hallway and Hallway2 POMDPs and compare the
  resulting rewards with those from the PBVI paper. This might turn up defects
  in my implementation or performance problems.
- Fix any defects or performance problems.
- Turn this into a Python package and upload to PyPi.


Known problems
--------------

### ValueError: zero-size array to reduction operation minimum which has no identity

Brilliant exception from NumPy, isn't it?

Likely cause: you're trying to load `X.POMPD.json` with
`piglet_pomdp.json_pomdp.load_pomdp`, but the reward function defined by the
original `X.POMDP` depended not only on the action and state before the action.

Explanation: PBVI, as defined in the [article](#pbvi-article), solves POMDPs
where the reward function depends only on the action and the state before the
action, R(st, at). When you use `pomdp2json.py` to convert a file that doesn't
stick to that restriction, it will produce a POMDP.json file that is
incompatible with piglet_pomdp. (To detect incompatible input files and warn the
user is on the to-do list.) piglet_pomdp misbehaves when it receives
incompatible input.

Solution: In general, there is no easy way to convert a POMDP with a R(st, at,
st+1, ot+1) reward function to a POMDP with a R(st, at) reward function. There
might be a sophisticated way that I don't know. But there are also easy cases
where you can fiddle with the original POMDP file. For example, `hallway.POMDP`
defines this reward function:

```
# Rewards
# (R: <action> : <start-state> : <end-state> : <observation> %f)
R: * : * : 56 : * 1.000000
R: * : * : 57 : * 1.000000
R: * : * : 58 : * 1.000000
R: * : * : 59 : * 1.000000
```

If you swap columns like this:

```
R: * : 56 : * : * 1.000000
R: * : 57 : * : * 1.000000
R: * : 58 : * : * 1.000000
R: * : 59 : * : * 1.000000
```

Then the POMDP will be compatible with piglet_pbvi. Is it the same as the
original? Maybe not. But I guess it's almost the same.


Why?
----

- Why is the package called `piglet_pbvi`? Because I want to avoid name clashes
  with other people's PBVI implementations. ‘PBVI’ is also used as a generic
  term for point-based POMDP solution methods. So I took the first letters of
  the family names of the [authors of the paper](#pbvi-article) that describes
  the PBVI that *I'm* implementing and added an i and an l and an e to aid your
  memory.


License
-------

See [LICENSE.txt](LICENSE.txt).

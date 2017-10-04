Implementation of Point-Based Value Iteration
=============================================

Point-based value iteration (PBVI) is an approximate method for solving
partially observable Markov decision processes (POMDPs). I'm implementing the
algorithm from:

[Joelle Pineau, Geoffrey Gordon, Sebastian Thrun. Point-based value iteration:
An anytime algorithm for POMDPs. In IJCAI,
2003.](http://ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf)


Done
----

- Implement PBVI.
- Implement a naive version that is slower, but easier to read.
- Test it on a two-state problem from AIMA. Compare value function with AIMA. –
  Looks good.
- Test it on the [Tiger95](http://www.pomdp.org/examples/) problem. Compare
  value function with the result of
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
  function will be limited to depend on the start state and action.
- Run my implementation on the Hallway and Hallway2 POMDPs and compare the
  resulting rewards with those from the PBVI paper. This might turn up defects
  in my implementation or performance problems.
- Fix any defects or performance problems.

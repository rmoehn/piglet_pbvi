Implementation of Point-Based Value Iteration
=============================================

Point-based value iteration (PBVI) is an approximate method for solving
partially observable Markov decision processes (POMDPs). I'm implementing the
algorithm from:

<a name="#pbvi-article">
[Joelle Pineau, Geoffrey Gordon, Sebastian Thrun. Point-based value iteration:
An anytime algorithm for POMDPs. In IJCAI,
2003.](https://www.ri.cmu.edu/pub_files/pub4/pineau_joelle_2003_3/pineau_joelle_2003_3.pdf)
</a>


Usage
-----

For trying out piglet_pbvi I recommend downloading the [Tiger
POMDP](http://www.pomdp.org/examples/tiger.95.POMDP) and [converting it to
JSON](https://github.com/rmoehn/pomdp2json).

```bash
# Prerequisites – See README of pomdp2json for details.
$ git clone https://github.com/rmoehn/pomdp2json
$ cd pomdp2json
$ ./build.sh
$ mkdir ../pomdp_defs
$ wget --directory-prefix=../pomdp_defs http://www.pomdp.org/examples/tiger.95.POMDP
$ python2.7 pomdp2json.py ../pomdp_defs/tiger.95.POMDP ../pomdp_defs/tiger.95.POMDP.json
$ cd ..

$ git clone https://github.com/rmoehn/piglet_pbvi
$ cd piglet_pbvi
$ python2.7 piglet_pbvi/json_pomdp.py ../pomdp_defs/tiger.95.POMDP.json
```

You can do the last step in Python as well:

```python
# Let Python know where to find the module, because I haven't made an
# installable package.
import sys
sys.path.insert(0, <path to piglet_pbvi repo clone>)

from piglet_pbvi import json_pomdp

pbvi_gen = json_pomdp.load_pomdp("<path to tiger.95.POMDP.json>")
for __ in xrange(10):
    print next(pbvi_gen)
    # Outputs current value function estimate and optimal policy for that value
    # function.
```

You might also have a look at [pomdp_play.py](pomdp_play.py). It's messy, but it
runs piglet_pbvi and naive piglet_pbvi on an adaptation of the POMDP defined in
[AIMA](http://aima.cs.berkeley.edu/).

I will add more documentation after I've made sure that piglet_pbvi works
correctly.


Disclaimer
----------

The implementation is far from mature. It probably has bugs and consists mostly
of undocumented NumPy oatmeal that I don't completely understand, even though
I've written it. It might not work for anything but the Tiger and AIMA POMDPs
mentioned above.


Done
----

- Implement [PBVI](piglet_pbvi/pbvi.py).
- Implement a [naive version](piglet_pbvi/naive_pbvi.py) that is slower, but
  easier to read.
- [Test](pomdp_play.py) them on a two-state problem from AIMA. Compare value
  function with AIMA. – Looks good.
- Test it on the [Tiger95](http://www.pomdp.org/examples/tiger.95.POMDP) POMDP.
  Compare value function with the result of
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
  - Convert README to reStructuredText.


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

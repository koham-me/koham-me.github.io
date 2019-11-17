---
layout: post
title: "Karger's Algorithm"
date: 2018-08-25
tags: [graph, algorithm, clustering]
comments: true

---

## Graph
A set of vertices V along with set of edges E connecting these vertices in is called graph and denoted as $$ G=(V,E) $$. Example include a social network where each individual can be considered as vertex and any kind of relationship between two individuals can be considered as an edge. Another example is road network connecting different cities. Below is an example of graph, each of vertex represents a city and edge represents flight connectivity.
![Flights]({{site.baseurl}}/img/flights.png)

## Minimum Cut
A cut in an undirected graph $$ G=(V,E) $$ is a partition of the vertices V into two non-empty, disjoint sets $$ S \cup T = V $$. Cutset of cut consits of the only edges connecting the vertcies in S to vertices in T. Number of edges in cutset is called size of cut. The cut with minimum size is called minimum cut. Each vertex can be in either in set S or set T, hence there are $$ 2^{|V|} $$ cuts. Out of these two sets are empty, also interchanging S and T does not make any difference. Hence total valid cuts are $$ 2^{|V|-1}-1 $$. We need to find cut with minimum size among all cuts.

## Edge Contraction
Edge contraction is method to remove an edge from graph and merge its two incident vertcies. For example if we remove edge Delhi-Chandigarh, then new graph will look like

![FlightsDC]({{site.baseurl}}/img/flightDC.png)
Please note that the all the edges except contracted edge incident on either of Delhi or Chandigarh will now be incident on new vertex DelhiC. In this manner we can continue contracting as many eddges as we want
and get a new graph. We allow here multiple incident edges between two vertices, such graph is called multigraph.

## Algorithm
Karger's algorithm uses edge contraction to find minimum cut. It selects an edge randomly, and perform edge contraction. This is continued untill two vertices are left. The vertices merged into final two vertices give minimum cut. For example below S = {Madurai, Trichy, Kochi, Chennai} merged into ChennaiMTK and T = {Chandigarh, Lucknow, Varanasi, Delhi} merged into DelhiCLV vertex. Number of edges in final graph gives the size of cut.
![FlightsDC]({{site.baseurl}}/img/contracted.png)

This algorthm is example of randomized algorithm which does random operations to achieve the solution.

## Analysis
We do probabilistic analysis of algorithm finding the correct solution. Let degree(v) is number of edges incident on vetex v. Then since each edge is counted twice, once on each of incident edges,hence sum of degress of all vertices is

$$
\sum_{v \in V} degree(v) = 2 \left| E \right |
$$

Average or expected degree of a vertex is given as

$$
Exp[degree(v)] = \sum_{u \in V} Pr(v=u) degree(u)
$$


$$
= \sum_{u \in V} \frac{1}{\left|V\right|} degree(u)
= \frac{1}{\left| V \right|} \sum_{u \in V} degree(u)
$$

$$
= \frac{2\left| E \right|}{\left| V \right|}
$$

Consider cut for V into S and T, where S contains only one vertex u and T contains all other vertices. Size of this cut will be same as degree(u). Since minimum cut can not be greater than this, we can say 

size of minimum cut <= degree(u)

In other words minimum cut is also less than or equal to average degree of vertices in G. Hence size of minimum cut is atmost

$$\frac{2|E|}{|V|} $$


There are maximum $$\frac{2\|E\|}{\|V\|}$$ edges in minimum cut, selecting an edge from minimum cut out of total has probability at most $$\frac{2}{\|V\|}$$. If we pick an edge randomly, probability it will be in cut have probability at most $$\frac{2}{\|V\|}$$. If we continue with the algorithm and do edge contraction each time, the probability that edge from cut will not be contracted, which is same as final result to be minimum cut is  

$$ (1 - \frac{2}{|V|}) (1 - \frac{2}{|V|-1} ) (1 - \frac{2}{|V|-2}) (1 - \frac{2}{|V|-3}) .. (1-\frac{2}{4})(1-\frac{2}{3})
$$

$$ = (\frac{|V|-2}{|V|})(\frac{|V|-3}{|V|-1}) .. (\frac{2}{4}) (\frac{1}{3})
$$

$$= \frac{2}{|V|(|V|-1)}$$

Hence algorithm succeeds with probability $$\frac{2}{n^2}$$. We can repeat the algorithm for $$n^2$$ times and chose the smallest cut.

This algorithm can be generalized to stop at any number of vertices. We can stop edge contraction reaching specified number of vertices. Which can be useful in community detection in social network, where each merged vertex can be a community.

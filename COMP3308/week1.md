# Key Dates

- [ ] (4%) Weekly Homework 
        Individual
     Est. Given: 3pm Tuesdays
            Due: 3pm Tuesdays 

- [ ] (12%) Assignment 1 
        Group
     Est. Given: 08 Mar 2025 Week 3 Friday
            Due: 08 Apr 2025 at 23:59 Week 7 Tuesday

- [ ] (24%) Assignment 2 
        Group
     Est. Given: 09 Apr 2025 Week 7 Wednesday
            Due: 09 May 2025 at 23:59 Week 10 Friday

- [ ] (60%) Final Exam 
        Individual
     Est. Given: Formal exam period 
            Due: Formal exam period

## Course Overview
- Problem Solving and Search
    - Path finding (Maps)
- Game Playing
    - Chess
- Machine Learning
    - Classification Example
    - Clustering Example
- Neural Networks and Deep Learning
- Probabilistic Reasoning and Inference
- Unsupervised Learning

### Misguided Assumption of Artificial Intelligence
Artificial Intelligence does not only consist of Generative Models.
An example is as follows:
> The program inside google maps to determine the fastest route
> to the destination can be considered an Artificial Intelligence.

#### 4 Approaches of Artificial Intelligence
``` txt
+-------------+-------------+
| Think like  | Think       |
| humans      | rationally  |
+-------------+-------------+
| Act like    | Act         |
| humans      | rationally  |
+-------------+-------------+
```

##### Turing Test (Act like humans)
The attempt to mimmick humans, via philosophical determination 
of the ability to "think".

Human interrogator asks written questions to two recipients.
It is unknown which recipient is a machine or human.

If the interrogator is unable to determine which recipient is
the machine, the machine can be considered "intelligent" (has
the ability to "think")

To pass the Turing Test, a computer needs:
- Natural Language Processing
    - Communicated successfully in (written) English
- Knowledge Representation
    - Store what it knows and hears
- Automated Reasoning
    - Use stored information to answer questions and 
      draw conclusions
- Machine Learning
    - Adapt to new situations and can detect new patterns
      and apply the to new situations

"Total" Turing Test will also need:
- Computer Vision
    - To percieve objects
- Robotics
    - To move around

[!warning] Takeaway
> It is more important to understand the principle behind
> intelligent behaviour and use them to build intelligent
> systems.

[!important] Analogy
> When humans attempt to fly, which process had more success?
> A. Atttempting to imitate how bird fly by flapping wings
> B. Attempting to learn basic aerodynamic principles 
>
> Answer: B, This can be observed in how aeroplanes are designed

##### The Cognitive Model Approach (Think like humans)
When we have a theory on how humans think, we can write a computer
program that follows this theory.
The theory is developed via:
- Introspection
- Psychological Experiments
- Brain imaging

##### The "Laws of Thought" approach (Think rationally)
Using logic to build intelligent systems.
    - takes a description of a problem in logical notation
    - determines the soltuion using correct inference
e.g.
    socrates is a man. All man are mortal.
    => Socrates is mortal

However, taking informal knowledge and representing it in formal
notation is difficult. And unable to handle probabilistic inference.

##### The Rational Agent Approach (Act rationally)
The currently use approach in AI.

- A Intelligent Agent:
    - Has in-built knowledge and goals
    - Percieves the environment
    - Acts rationally to achieve its goals by using its knowledge
      and current perception of the environment
    - Acting rationally = taking an action that maximizes performance

###### Model of Artificial Neuron
```
inputs         Output
  X -
      \
  X --- +---+
        |   |------
  X---- +---+
      /
  X -
```


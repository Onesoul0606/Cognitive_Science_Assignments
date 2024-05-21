# imports required for all tasks

# import numpy, the python scientific package
import numpy as np

# test images from scipy
from scipy import misc

# import maplotlib to plot the visualizations
from matplotlib import pyplot as plt

# 1. Simple neuron models

# simulation parameters

# simulation time (in milliseconds)
t_max = 200
# generate an input current
np.random.seed(42)
I = np.random.normal(size=t_max)
# Spike threshold
threshold = 0.75

# simulate model 1

# Adjusted the code to mark the point where it crosses the threshold
V1 = np.zeros(t_max)
spikes1 = []
reset_voltage = 0 # Performs the same role as V1[t]=0
for t in range(1, t_max):
    # If a spike occurred at the previous time step, consider the previous value as 0 for the calculation
    if t-1 in spikes1:
        V1[t] = reset_voltage + 1/10 * (-reset_voltage + 4.46*I[t])
    #If no spike occurred, proceed with normal calculation
    else: 
        V1[t] = V1[t-1] + 1/10 * (-V1[t-1] + 4.46*I[t])
    
    if V1[t] > threshold:
        V1[t] = threshold
        spikes1.append(t)


# simulate model 2
V2 = I
spikes2 = np.where(V2>threshold)[0]

# plot the results

#To align the actual time progression with the graph
time_axis = np.arange(1, t_max+1)

# plot the input current
plt.figure(figsize=(16,7))
plt.subplot(311)
plt.plot(time_axis, I, color='orange')
plt.xlim(0,t_max)
plt.xticks(())
plt.ylabel('Current (a.u.)')

plt.subplot(312)
plt.plot(time_axis, V1, label='model 1', color='blue')
plt.plot(time_axis, V2, label='model 2', color='orange')
plt.scatter(np.array(spikes1) + 1, V1[spikes1], color='red', s=20)
plt.scatter(np.array(spikes2) + 1, V2[spikes2], color='green',marker='x', s=20)
plt.hlines(threshold, 0,t_max, color='grey', ls='--', lw=1, label='Threshold')
plt.xlim(0,t_max)
plt.xticks(())
plt.ylabel('Voltage (a.u.)')
plt.legend(loc='upper right', fontsize=8)

# To align the actual time progression with the graph
plt.subplot(313)
plt.scatter(np.array(spikes1) + 1, np.zeros(len(spikes1)), marker='|', color='blue', s=500, label='model 1')
plt.scatter(np.array(spikes2) + 1, np.ones(len(spikes2)), marker='|', color='orange', s=500, label='model 2')
plt.xlim(0,t_max)
plt.ylim(-.5,1.5)
plt.yticks(())
plt.xlabel('Time (ms)')
plt.legend()
plt.show()

'''
Exercise 1a)
Explain how these models work and state the equations.

This code simulates two neuron models and compares them. Model 1 emulates the Leaky Integrate and Fire (LIF) model.

The previous V1 Model's code did not display the time step when the threshold was exceeded, so it was 
to make the graph easier to read by showing the threshold-crossing moment.

In Model 1, the voltage array V1 is started with zeros for the range of the simulation _[V1 = np.zeros(t_max)]_ , and an empty list 'spikes1' is prepared to score the spike times _[spikes1 = [ ]]_. The simulation iterates each time step starting from 1 to 200 milliseconds _[for t in range(1, t_max):]_. If a spike occurred at the previous time step _[if t-1 in spikes1:]_, consider the previous value as 0 for the calculation _[V1[t] = reset_voltage + 1/10 * (-reset_voltage + 4.46*I[t])]_. Else if no spike occurred, proceed with normal calculation with the voltage at each moment based on the previous voltage and the input current at that time, multiplied by a predefined scale factor _[V1[t] = V1[t-1] + 1/10 * (-V1[t-1] + 4.46*I[t]) ]_. If the voltage at any time exceeds a set threshold(0.75) _[if V1[t]>threshold:]_ , the time of the spike is recorded_[spikes1.append(t)]_ , and the voltage is reset to 0 _[reset_voltage = 0]_.

Equation for $V_1$:

$V_1[t] = V_1[t-1] +  \frac{1}{10}(-V_1[t-1] + 4.46 \cdot I[t])$ 

where:
- $V_1[t]$ corresponds to the membrane potential stored in the neuron at the current time step.
- $V_1[t-1]$ corresponds to the membrane potential stored up to the previous time step.
- The numerator of the fraction(1) corresponds to the time step. Since this model measures every 1 millisecond, it is represented by 1.   In the formula, this is expressed as $\Delta_t$.
- The denominator of the fraction(10) corresponds to the membrane time constant. This is expressed as $\tau_m$ and the formula of time constant is $\tau_m = R_mC_M$ ($R_m :$ membrane resistance, $C_m :$ membrane capacitance).
    - Why 10?: The time constant is defined as the amount of time it takes for the change in potential to reach 63.2% of its final value, which is 10 milliseconds.
- $-V_1[t-1]$ corresponds to the leak of the membrane potential. This indirectly represents the leaky effect caused by subtracting the resting potential from the membrane potential $(V(t) - V_{rest})$.
- 4.46 corresponds to resistance of the membrane. In the formula, this is expressed as $R$. In this model, it is used as coefficient to make V1 and V2 have roughly the same variability (in the absence of spikes)
- $I[t]$ corresponds to the external electrical signal input to the neurons at each time step. 

In model 2, V2 directly sets be equal to the external electrical signal input 'I' _[V2 = I]_. This model Identifies the time at which the voltage exceeds the threshold _[spikes2 = np.where(V2>threshold)[0]]_ .

Equation for $V_2$:

$V_2 = I$

Therefore, according to the hypothesis I've made, the significance of this model lies in emulating the generation of electrical signals (action potentials, spikes) in pre-synaptic neurons(model 2) in order to send chemical signals to post-synaptic neurons(Model 1) in the neuron system. It is a modeling that formalises the process where the membrane potential generates a spike and resets to the reset voltage when it exceeds the threshold voltage, while a continuous leak occurred.
'''

'''
Exercise 1b)
Explain how the spiking differs between the two models, and explain which components of the model cause those differences.

The key difference in spiking between the two models stems from how each model processes the input current and generates voltage responses through different mechanisms.

Model 1:

Time-based integration: Model 1 integrates the input current over time, which implies gradual change of the voltage. This mimics the process by which actual neurons accumulate and process continuous stimuli over time. In the equation, it corresponds to  𝑉1[𝑡]=𝑉1[𝑡−1]+...
 , which means the current membrane potential is calculated by sum of previous potential membrane and newly computed external input.

Leaky effect: The neuron's voltage tends to naturally decrease, reflecting the membrane potential of actual neurons. The leak indicates that neurons require continuous and sufficient magnitude of external current input to maintain an active state.In the equation, it corresponds to  110(−𝑉1[𝑡−1]+4.46⋅𝐼[𝑡])
 .

Threshold and voltage reset: When the voltage exceeds a certain threshold(0.75 in this model), a spike occurs, and the voltage is reset to the resting voltage . It represents the refractory period during which a neuron is not activated for a certain time after generating a spike.

To sum up, the Model 1 integrates the input current over time and allows the voltage to naturally decrease through its leaky characteristic. In this model, a spike occurs when the voltage exceeds a threshold, and the voltage is immediately reset. This process mimics the behavior of real neurons, where the neuron's response to input accumulates over time and generates a spike when it reaches the threshold.

Model 2:

Model 2, like Model 1, produces spikes as soon as the threshold is exceeded. However, it directly uses the input current as the voltage, ommiting the processes of temporal integration or the leaky effect on the membrane potential. This means spike generation is entirely reliant on instantaneous changes in the input current, leading to potentially more frequent spiking compared to Model 1.

In other words, Model 2 acts as a pre-synaptic neurons, skipping the detailed description of the action potential process modeled in Model 1. This means Model 2 appears to simply perform the role of sending spiking signals to Model 1, which acts as a post-synaptic neurons.

In conclusion, Model 1 represents the neuron's gradual response to input through temporal integration and leaky characteristics, reflecting a more realistic neuronal behavior that generates spikes only upon reaching the threshold. In contrast, Model 2 generates spikes immediately when the input current exceeds the threshold, omitting the real neuron's dynamic features such as the temporal integration process, and leaky effect.
'''

'''
Exercise 1c)
Explain which aspects of the activity of real neurons are captured by each model, and which are missing.

The electrical conduction in an actual neurons go through four main states to generate a spike: 1. Resting potential, 2. Depolarisation, 3. Repolarisation, 4. Hyperpolarisation (undershoot).

Resting Potential
Captured by the model:
The model reflects the cycle of an action potential where it starts at resting potential, is stimulated to generate an action potential, and then returns to resting potential. Model 1 starting with an initial value of 0 and resetting to 0 after spiking represents one cycle of an action potential.
Missed by the model:
The actual neuron’s resting potential is not 0 but typically -70mV, creating a numerical discrepancy. However, it's presumed this simplification to 0 was made intentionally for model simplification.
Depolarisation, Repolarisation
Captured by the model:
In real neurons, action potential rapidly increases to +30mV within about 1ms when external stimuli reach a certain threshold. Both Model 1 and 2 replicate this by producing a spike and quickly stabilising once the threshold is exceeded.
Real neurons operate on an all-or-none law, responding uniformly to stimuli above a certain threshold. Models 1 handles the moment when an action potential occured by recording the point at which spiking occurs as soon as the threshhold is exceeded.
Captured by the model:
The graph of model1 shows the moment of spiking and the stabilisation of the action potential, but it only represents up to the point whrer it exceeds the threshhold, ommitting to describe the rapid rise of the action potential.
Hyperpolarisation
Missed by the model:
The neuron’s membrane potential drops below the resting potential after repolarisation, which is not represented in this model.
Refractory Periods
Captured by the model:
There are absolute and relative refractory periods. The absolute refractory period occurs during an action potential when no amount of additional stimulus has any effect. The relative refractory period, usually referring to the hyperpolarisation state, is when the membrane potential is lower than resting potential and requires a stronger-than-usual stimulus to generate another action potential. This model effectively implements the absolute refractory period by recording a spike and resetting immediately upon reaching the threshold.
Missed by the model:
The neuron's refractory period varies with different factors, which the model simplifies to a fixed time post-voltage reset. Additionally, the implementation of the relative refractory period was not addressed.
Synaptic Transmission
Missed by the model:
The model does not reflect the chemical transmission of stimuli via neurotransmitters from the pre-synaptic to the post-synaptic neuron, overlooking the signal's type.
'''


#2. Receptive fields in the visual system

x = np.tile(np.arange(-8,9),(17,1))
y = x.T 

# the first model
s1=3
kernel1 = -np.exp(-(x*x+y*y)/(2*s1**2))*np.sin(y/4)  # do not change this line
kernel1_1 = -np.exp(-(x*x+y*y)/(2*s1**2))
kernel1_2 = np.sin(y/4)

# the second model
s1=3
s2=4
kernel2 = np.exp(-(x*x+y*y)/(2*s1**2)) - 0.5*np.exp(-(x*x+y*y)/(2*s2**2)) # do not change this line
kernel2_1 = np.exp(-(x*x+y*y)/(2*s1**2))
kernel2_2 = 0.5*np.exp(-(x*x+y*y)/(2*s2**2))

# the third model
s1=3
kernel3 = np.exp(-(x*x+y*y)/(2*s1**2))*np.cos((x.T/1.25+np.cos(np.pi/3)*y[:,0])) # do not change this line
kernel3_1 = np.exp(-(x*x+y*y)/(2*s1**2))
kernel3_2 = np.cos((x.T/1.25+np.cos(np.pi/3)*y[:,0])) 

# plot the matrices
plt.figure(figsize=(16,12))
for i,kernel in enumerate((kernel1,kernel2,kernel3)):
    plt.subplot(3,3,3*i+1)
    maxv = np.max((np.abs(np.min(kernel)), np.max(kernel)))
    plt.imshow(kernel, cmap=plt.cm.RdBu_r, vmin=-maxv, vmax=maxv)
    plt.colorbar()
    plt.title('receptive field {}'.format(i+1))
                    
for i,kernel in enumerate((kernel1_1,kernel2_1,kernel3_1)):
    plt.subplot(3,3,3*i+2)
    maxv = np.max((np.abs(np.min(kernel)), np.max(kernel)))
    plt.imshow(kernel, cmap=plt.cm.RdBu_r, vmin=-maxv, vmax=maxv)
    plt.colorbar()
    plt.title('receptive field {}_1'.format(i+1))
                    
for i,kernel in enumerate((kernel1_2,kernel2_2,kernel3_2)):
    plt.subplot(3,3,3*i+3)
    maxv = np.max((np.abs(np.min(kernel)), np.max(kernel)))
    plt.imshow(kernel, cmap=plt.cm.RdBu_r, vmin=-maxv, vmax=maxv)
    plt.colorbar()
    plt.title('receptive field {}_2'.format(i+1))
    
from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(np.arange(0, 17), np.arange(0, 17))

# Plotting
fig = plt.figure(figsize=(16, 12))
for i, kernel in enumerate((kernel1, kernel2, kernel3)):
    ax = fig.add_subplot(3, 1, i + 1, projection='3d')
    maxv = np.max((np.abs(np.min(kernel)), np.max(kernel)))
    surf = ax.plot_surface(X, Y, kernel, cmap=plt.cm.RdBu_r, vmin=-maxv, vmax=maxv)
    ax.view_init(azim=10,elev=20)
    fig.colorbar(surf, ax=ax)
    ax.set_title('Receptive Field 3D {}'.format(i + 1))

'''
Exercise 2a)
Describe the three receptive fields shown above. State which cell type each receptive field best matches, and where in the visual system these cells are found. Which feature in an image most strongly excites each cell type, and which features yield the weakest response?

Model 1 Characteristics:

This model is based on Gabor filters, combining a negative exponential gaussian function that varies with position (x,y) ( −𝑛𝑝.𝑒𝑥𝑝(−(𝑥∗𝑥+𝑦∗𝑦)/(2∗𝑠1∗∗2)))
  and a sine harmonic function in the y orientation  (𝑛𝑝.𝑠𝑖𝑛(𝑦/4))
 . The first function restricts the area affected by the filter and decreases its influence as it moves away from the centre. The second function aids the filter in detecting edges or bars in a specific orientation.Therefore, this models the response that occurs when neurons detect a line in a specific orientation.In the resulting image, the horizontal structure of the image is emphasised, indicating that the receptive fields are sensitively designed to changes in the specific orientation. Both positive (red) and negative (blue) responses are observed, signifying that neurons are activated in areas where orientational edges or bars exist.
In the Visual System:

The best matching cell type for this model is simple cells, which are located in the Primary Visual Cortex (V1). The results from the first model align with the characteristics of simple cells specialised in detecting linear edges.
Strongest and Weakest Features:

The strongest response is is observed when edges or bars align with the range and orientation of the receptive field. The weakest response is observed with edges or bars that do not match the orientation or in areas of uniform brightness. Additionally, it does not distinguish dynamic directions.
Model 2 Characteristics:

This model is based on the Difference of Gaussians (DoG) and models how neurons respond to small points of light or stimuli of a specific size. The first term represents the gaussian function with a standard deviation of 's1'  (𝑛𝑝.𝑒𝑥𝑝(−(𝑥∗𝑥+𝑦∗𝑦)/(2∗𝑠1∗∗2))−0.5)
  and the second term represents the gaussian function with a standard deviation of 's2'  0.5∗(𝑛𝑝.𝑒𝑥𝑝(−(𝑥∗𝑥+𝑦∗𝑦)/(2∗𝑠2∗∗2)))
 . By subtracting a larger standard deviation 's2' from a smaller standard deviation 's1', it's possible to specify a narrow range. Therefore, in the resulting image, a circular area in the narrow center is emphasised, indicating it is sensitively designed to stimuli within a certain range. A positive response (red) is predominantly observed, meaning that neurons are activated only within a specific range.
In the Visual System:

The best matching cell type for this model is the Retinal Ganglion Cells (RGCs), which are located downstream of the Lateral Geniculate Nucleus (LGN) of the actual visual system. The results of the second model align with the characteristics of Retinal Ganglion Cells that have On-centers OFF-surrounds features.
Strongest and Weakest Features:

This receptive field exhibits a center-surround structure, meaning it responds most strongly when small objects or points of light stimulate the central part. Conversely, the weakest response is observed with large uniform stimuli or when both the center and surround are stimulated simultaneously, causing an inhibitory effect.
Model 3 Charactersistics:

This model is also based on Gabor filters. The first term performs the same role as Model 1. This function is utilized to detect patterns with specific orientations and frequencies in an image.  𝑇/1.25
  represents the component in the x direction, and  𝑐𝑜𝑠(π/3)⋅𝑦[:,0]
  adjusts the component in the y direction. Notably,  𝑦[:,0]
  indicates the first column of y, a notation that can be used in multidimensional arrays. The composition of this formula shows that the filter is designed to sensitively respond to specific directions and frequencies. Therefore, in the resulting image, the diagram responds strongly to stimuli of a certain orientation, regardless of positional characteristics, in both positive and negative responese, meaning that ,similar to a simple cell, this model mainly reacts to oriented edges and patterns but possesses some level of position invariance.
In the Visual System:

The best matching cell type for this model is complex cells, which are located in the Primary Visual Cortex (V1).The results of the third model align with the characteristics of complex Cells that receive information from simple cells and have the features of responding when moving in a certain direction, regardless of spatial position.
Strongest and Weakest Features:

Complex cells can recognise an object even when its position changes and are better at identifying edges or textures in various directions and sizes. Paradoxically, the spatial invariance of complex cells can be a weakness in recognizing fine details. They may be less sensitive than simple cells in distinguishing the clear edges, changes in specific positions or subtle patterns.
'''


#Exercise 2b)
#Now test your intuition on what type of filtering these receptive fields perform by computing the responses of these neurons to a natural image. To this end we assume we have as many neurons as pixels in the image, and the centre locations of the receptive fields match the pixel locations. Then we can use a simple convolution operation to predict the response of each neuron.
#Show the predicted responses for each receptive field. Use an appropriate colour scale (use a diverging colormap with the zero clearly visible, as above), to indicate positive and negative responses.

# here we use the 2D convolution function in scipy
from scipy.signal import convolve2d

# The test image as a 2D matrix:
image = misc.ascent()

# to see the original image, use:
plt.figure(figsize=(16,3))
plt.subplot(1,4,1)
plt.imshow(misc.ascent(),cmap=plt.cm.gray)

# the result for the first receptive field
conv = convolve2d(image,kernel1)

# Convolution for each receptive field
conv1 = convolve2d(image, kernel1)
conv2 = convolve2d(image, kernel2)
conv3 = convolve2d(image, kernel3)

plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
plt.imshow(misc.ascent(),cmap=plt.cm.gray)
plt.title('Original Image')
for i,convolution in enumerate((conv1,conv2,conv3)):
    plt.subplot(2,2,i+2)
    maxv = np.max((np.abs(np.min(convolution)), np.max(convolution)))
    plt.imshow(convolution, cmap=plt.cm.RdBu_r, vmin=-maxv, vmax=maxv)
    plt.colorbar()
    plt.title('Response of Receptive Field {}'.format(i+1)) 
    
'''
Exercise 2c)
Describe and explain your results. Are the receptive fields indeed selective to the image features you predicted above? Why do neurons in the visual system respond in this way?

Why do neurons in the visual system respond in this way?

This behaviour of neurons in the visual system is thought to mirror the sequence of importance in our visual information processing. Essentially, the modeled pathway mimics the natural visual system, starting with the LGN (model 2), which processes the brightness of the light source due to the fundamental need for light in visual perception. This is followed by the simple cells of V1 (model 1), which identify the edges of objects to capture their basic outlines and simple characteristics. Finally, the complex cells of V1 (model 3) evolve to discern intricate details, such as object patterns and movement direction. This hierarchical structure reflects the system's adaptation to efficiently process visual information by prioritising the analysis of light, shape, and detail in that order.
Describe and explain your results

As previously mentioned, Model 2, which is positioned at the beginning of the visual pathway, analyses images primarily based on the brightness of light sources. Given that the initial image is a black and white photograph, this model responds in the positive way(red) to areas that are closer to white, ignoring the distribution of shadows or the true color of the objects. Conversely, it does not react to areas that are closer to black.

Model 1 interprets images based on the distinct edges of objects. Based on the original photo, it does not respond to backgrounds of uniform brightness or vertical lines. Additionally, the criteria for positive and negative values in Model 1 can be interpreted based on Model 2. The basis for the lines' upper and lower parts was set as follows: assuming that the orientational edge lines recognized by Model 1 move anti-clockwise from  𝜋
  to  2𝜋
  (180 to 360 degrees), based on the negative exponential gaussian function, in Model 1 which also makes the sine harmonic function  −𝑠𝑖𝑛
 . Thus, The part of the edge already passed is defined as 'bottom', while the part yet to be traversed is defined as 'top'. Consequently, edges with brighter tops (based on Model 2, if the top is more red) had negative (blue) lines drawn, and edges with darker tops (based on Model 2, if the top is more white) had positive (red) lines drawn. The clarity of the color depended on the difference in brightness of the 'top' and 'bottom'.

Model 3 integrates information from Model 1 to a higher level, analysing more complex visual features like movement or patterns such as diagonal patterns on railings, horizontal pillars, and stair handrails. Furthermore, as mentioned above, being position-invariant means that it consistently responds to movements in the same direction, even if the stimuli move within the visual field. Therefore, although Model 3 may not distinguish edges as clearly as simple cells, it discerns patterns, textures, and directions in more refined areas. This indicates that it captures more complex structural information, creating a sense of visual space.

Are the receptive fields indeed selective to the image features you predicted above?

It was clearly observed that the receptive fields in all models selectively responded to the image features as predicted.
'''

#3. A memory model
'''
Here you will simulate the Hopfield model. This model simulates  𝑀
  binary neurons with the following activity rule:

𝑠𝑖(𝑡+1)Θ(𝑎)==Θ(∑𝑗=1𝑀𝑤𝑖𝑗𝑠𝑗(𝑡)−𝜃𝑖){10𝑎≥0𝑎<0
 
The variable  𝑠𝑖(𝑡)
  is the activity of neuron  𝑖
  at time  𝑡
 , and  𝑤𝑖𝑗
  is the weight between neurons  𝑖
  and  𝑗
 . In the following we set the bias  𝜃𝑖=0
 .

The following learning rule is used to store a set of  𝑁
  binary patterns  𝑝𝑛𝑖∈{0,1}
  in this network:

𝑤𝑖𝑗=1𝑁∑𝑛=1𝑁(𝑝𝑛𝑖−𝑠)(𝑝𝑛𝑗−𝑠)
 
The quantity  𝑠
  is the sparseness of the patterns, which is the fraction of elements in the patterns that are 1. So if a pattern of length 20 contains 10 times the '0' and 10 times '1',  𝑠=0.5
 .

In other words, sparseness is the arithmentic mean (average) over all patterns:  𝑠=<𝑝𝑛>𝑁.
'''
# Let's create a binary pattern to store, here we use a racoon:
image = misc.face(gray=True)[:600:10,280:950:10]>140
pattern = image.flatten().astype(int)

plt.figure(figsize=(8,3))
plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)

# We also use the mirror image of the same racoon
image2 = image[:,::-1]
pattern2 = image2.flatten().astype(int)

plt.subplot(122)
plt.imshow(image2, cmap=plt.cm.gray);

'''
Exercise 3a)
Briefly describe in words what this program computes. Address the code in parts 1, 2 and 3 in turn in your description. Do not yet consider the output. What type of memory is simulated here?

This program simulates the storage and recovery process of binary patterns using a Hopfield network model. Hopfield network model is based on associative memory system. Associative memory system combines corrlated information or patterns with a given input to produce a result. A characteristic feature of associative memory is:

1) Recovery of the complete pattern from partial input: Associative memory systems can deduce and recover complete information from just a part of the input provided. 2) Error tolerance: This system has the capability to deliver correct outputs for inputs that are damaged or only partially provided. This means that it can accurately recall related information even if the input data has noise or some information is missing. 3) Pattern recognition: The system is capable of recognising patterns similar to those stored and recalling the corresponding information.

This code implements a Hopfield network, stores two patterns (the original raccoon image and its mirror image), and then simulates the process of recovering a partially damaged pattern using the network.

Part1: Weight initialisation and learning

1) The weight matrix w is initialised with random values  𝑤=𝑛𝑝.𝑟𝑎𝑛𝑑𝑜𝑚.𝑟𝑎𝑛𝑑𝑛(𝑀,𝑀)∗2
 . This corresponds to the initial state of the network's before starting to learn.
2) Weight is updated for the two patterns(pattern, pattern2) by outer product  𝑤+=𝑛𝑝.𝑜𝑢𝑡𝑒𝑟
 . During this process, weight is adjusted considering the sparsity of each pattern  (𝑝𝑎𝑡𝑡𝑒𝑟𝑛−𝑠𝑝𝑎𝑟𝑠𝑒𝑛𝑒𝑠𝑠,𝑝𝑎𝑡𝑡𝑒𝑟𝑛−𝑠𝑝𝑎𝑟𝑠𝑒𝑛𝑒𝑠𝑠),(𝑝𝑎𝑡𝑡𝑒𝑟𝑛2−𝑠𝑝𝑎𝑟𝑠𝑒𝑛𝑒𝑠𝑠,𝑝𝑎𝑡𝑡𝑒𝑟𝑛2−𝑠𝑝𝑎𝑟𝑠𝑒𝑛𝑒𝑠𝑠)
 .
3) the weight is averaged by dividing by the number of stored patterns  𝑤=𝑤/𝑁
 . This represents the adjustment process that the network enable to memorise the trained pattern equally.
Part 2: Damaging the pattern to set the Initial state

1) A copy of the pattern is made and stored to create a damaged version of this pattern  𝑠=𝑛𝑝.𝑐𝑜𝑝𝑦(𝑝𝑎𝑡𝑡𝑒𝑟𝑛)
 .
2) Corruption is created using random numbers from a standard normal distribution. Numbers greater than 0.7 are marked as 'True' in the corruption array, indicating which parts of the pattern will be damaged  𝑐𝑜𝑟𝑟𝑢𝑝𝑡=𝑛𝑝.𝑟𝑎𝑛𝑑𝑜𝑚.𝑟𝑎𝑛𝑑(𝑀)>0.7
 .
3) Elements from pattern2 are copied to s at positions marked as 'True' in the corrupt array. Consequently, approximately 30% of the original pattern s is replaced by corresponding parts from pattern2, resulting in damage  𝑠[𝑐𝑜𝑟𝑟𝑢𝑝𝑡]=𝑝𝑎𝑡𝑡𝑒𝑟𝑛2[𝑐𝑜𝑟𝑟𝑢𝑝𝑡]
 .
4) The Mean Squared Error (MSE) between the damaged pattern s and the original pattern is calculated  𝑝𝑟𝑖𝑛𝑡(𝑛𝑝.𝑠𝑢𝑚((𝑠−𝑝𝑎𝑡𝑡𝑒𝑟𝑛)∗∗2)/𝑙𝑒𝑛(𝑝𝑎𝑡𝑡𝑒𝑟𝑛))
 .
5) The image in its damaged state is visualised in the first subplot  𝑎𝑥[0].𝑖𝑚𝑠ℎ𝑜𝑤(𝑠.𝑟𝑒𝑠ℎ𝑎𝑝𝑒(𝑖𝑚𝑎𝑔𝑒.𝑠ℎ𝑎𝑝𝑒),𝑐𝑚𝑎𝑝=𝑝𝑙𝑡.𝑐𝑚.𝑔𝑟𝑎𝑦)
 .
Part 3: Recovering the pattern through the network model

1) The network is updated over several time steps 'T'  𝑓𝑜𝑟𝑡𝑖𝑛𝑟𝑎𝑛𝑔𝑒(𝑇)
 .
2) The current state s at each time step is calculated by subtracting the bias value from each neuron's net input signal and then applying an activation threshold.
3) The input signal is the result of subtracting the bias from the matrix multiplication of the weight matrix 'w' and the current state 's'  (𝑤@𝑠)−𝑏𝑖𝑎𝑠)
 , with positive input values returning 1, negative values returning -1, and zero returning 0  𝑛𝑝.𝑠𝑖𝑔𝑛()
 .
4) 0.5 is multiplied to the value that adjusts the range of returned values to -0.5, 0, 0.5  0.5∗
 . And 0.5 is added to the entire expression, the range of final result becomes 0, 0.5, 1  0.5+
 .
5) The calculated value is converted to an integer type that changes 0 and 0.5 to 0, and 1 remains 1, setting the neuron's state to either 0 or 1  .𝑎𝑠𝑡𝑦𝑝𝑒(𝑖𝑛𝑡)
 .
6) The whole formula is adjusted to produce a binary state of 0 or 1, updating the network's state s for the next time step.
7) The image in its damaged state is visualised in the each timesteps of subplot  𝑎𝑥[𝑡+1].𝑖𝑚𝑠ℎ𝑜𝑤(𝑠.𝑟𝑒𝑠ℎ𝑎𝑝𝑒(𝑖𝑚𝑎𝑔𝑒.𝑠ℎ𝑎𝑝𝑒),𝑐𝑚𝑎𝑝=𝑝𝑙𝑡.𝑐𝑚.𝑔𝑟𝑎𝑦)
 .
8) At each time step, the Mean Squared Error (MSE) between the recovered pattern 's' and the original pattern is calculated and printed  𝑛𝑝.𝑠𝑢𝑚((𝑠−𝑝𝑎𝑡𝑡𝑒𝑟𝑛)∗∗2)/𝑙𝑒𝑛(𝑝𝑎𝑡𝑡𝑒𝑟𝑛)
 .
'''

# parameters
N = 2 # number of patterns
M = pattern.shape[0] # number of neurons
sparseness = np.sum(pattern)/len(pattern)
bias = 0.7 # theta
bias2 = 0.2
T = 11 # number of time steps

# plotting bias 0.7 and weight 0.617
fig, ax = plt.subplots(T//6+1,6, figsize=(16,(T//6+1)*3))
ax = ax.flatten()

# part 1
w = np.random.randn(M,M)*2 # random initialisation
w += np.outer(pattern-sparseness, pattern-sparseness)
pattern2 = image[:,::-1].flatten().astype(int)
w += np.outer(pattern2-sparseness, pattern2-sparseness)
w = w/N

# part 2
np.random.seed(42)
s = np.copy(pattern)
corrupt = np.random.rand(M)>0.617 # weight
s[corrupt] = pattern2[corrupt]
print(np.sum((s-pattern)**2)/len(pattern))
ax[0].imshow(s.reshape(image.shape),cmap=plt.cm.gray)

# part 3
for t in range(T):
    s = (0.5 + 0.5 * np.sign((w @ s) - bias)).astype(int)
    ax[t+1].imshow(s.reshape(image.shape),cmap=plt.cm.gray)
    print(np.sum((s-pattern)**2)/len(pattern))
    plt.suptitle("Images of bias = 0.7 and weight = 0.617",fontsize=16)
    
# plotting bias 0.1 and weight 0.617
fig, ax = plt.subplots(T//6+1,6, figsize=(16,(T//6+1)*3))
ax = ax.flatten()

# part 2
np.random.seed(42)
s = np.copy(pattern)
corrupt = np.random.rand(M)>0.617 # weight
s[corrupt] = pattern2[corrupt]
print(np.sum((s-pattern)**2)/len(pattern))
ax[0].imshow(s.reshape(image.shape),cmap=plt.cm.gray)

# part 3
for t in range(T):
    s = (0.5 + 0.5 * np.sign((w @ s) - bias2)).astype(int)
    ax[t+1].imshow(s.reshape(image.shape),cmap=plt.cm.gray)
    print(np.sum((s-pattern)**2)/len(pattern))
    plt.suptitle("Images of bias = 0.2 and weight = 0.617",fontsize=16)
    
'''
Exercise 3b)
Now look at the output this code generates (text and graphs). How do you interpret these?

As explained above, the output of the code(text and graphs) represents the Mean Squared Error(MSE) before and after recovery at each time step. 
A lower MSE value indicates a smaller difference between the original pattern and the damaged pattern. 
It should be noted that although the ordinary MSE formula subtracts the predicted value(s) from the actual value(pattern), in this code, it's implemented the other way, with the predicted value(s) being subtracted from the actual value(pattern). 
However, since the difference is squared to calculate the MSE, it does not affect the numerical outcome, resulting in the same value. 
Returning to the explanation, part 2 created an initial pattern with noise. Since this code uses two patterns, the original (pattern) and its mirror image (pattern2), 
a value closer to 0 indicates similarity to the original pattern, and closer to 0.5 indicates similarity to the mirror image. Hence, the MSE value in the first image signifies the difference from the original pattern. 
Through part 3, the damaged pattern was perfectly restored to the original pattern. 
Initially, the changes were minimal, but after a specific time step(7~), it showed a rapid rate of recovery, suggesting the moment the model recognizes the original from the damaged pattern.

Consequently, the Hopfield network takes a damaged input and recovers it to one of the stored memory patterns, reducing the error to zero, which signifies that the network can accurately recall the stored pattern. This indicates that the model is highly effective in recovering damaged information, mimicking associative memory.
'''

'''
Exercise 3c)
What happens when you change the relative weight of the test pattern (the line marked with weight in the code)?

An increase in weight indicates a decrease in the proportion of 'True' values in 'corrupt', implying that as weight increases, the amount of noise decreases, bringing it closer to the original pattern. Thus, it becomes easier for this network model to recover the original pattern.

Conversely, a decrease in weight indicates an increase in the proportion of True values in corrupt, implying that as weight decreases, the amount of noise increases, making it closer to the mirror pattern (pattern2). Therefore, it becomes more challenging for the network model to recover the original pattern. When the weight drops to around 0.5, the original and mirror patterns become equally mixed, making it impossible to distinguish which image was original, starting the recovery with a blended image of the two patterns. Finally, when the weight is significantly lower than 0.5, the increase in the noise ratio leads to the model recognising the original of the initially damaged pattern as the mirror image, causing the MSE to rise and starting an inverse recovery process.

Therefore, the more elements of the pattern are damaged, the harder it becomes for the network to accurately recover the original pattern. Also, this causes the model to correct more errors in identifying the original pattern, and if too low, it may leads to misidentify the original pattern.

Additionally, referring to the graph above, since the bias represents the threshold for neuron activation, a higher number indicates a more conservative and less sensitive response to changes. However, when the weight value is definitive, meaning the 'True' ratio of corrupt is sufficiently biased towards one side, it was found that change in 'bias' does not cause significant variations. Consequently, the role of bias seems to be in fine-tuning the model's response sensitivity when the weight value is too ambiguous to clarify.
'''

# parameters
N = 2 # number of patterns
M = pattern.shape[0] # number of neurons
sparseness = np.sum(pattern)/len(pattern)
bias = 0.1 # theta
bias2 = 0.9
bias3 = 0.5
T = 5 # number of time steps

# sum of two images
image_sum = image + image2
plt.imshow(image_sum, cmap=plt.cm.gray);
plt.title("Sum of original image and mirror image")

# plotting bias = 0.1 and weight 0.7
fig, ax = plt.subplots(T//6+1,6, figsize=(16,(T//6+1)*3))
ax = ax.flatten()

np.random.seed(42)
s = np.copy(pattern)
corrupt = np.random.rand(M)>0.7 # weight
s[corrupt] = pattern2[corrupt]
print(np.sum((s-pattern)**2)/len(pattern))
ax[0].imshow(s.reshape(image.shape),cmap=plt.cm.gray)

for t in range(T):
    s = (0.5 + 0.5 * np.sign((w @ s) - bias)).astype(int)
    ax[t+1].imshow(s.reshape(image.shape),cmap=plt.cm.gray)
    print(np.sum((s-pattern)**2)/len(pattern))
    plt.suptitle("Images of bias = 0.1 and weight = 0.7",fontsize=16)

# plotting bias = 0.9 and weight 0.7
fig, ax = plt.subplots(T//6+1,6, figsize=(16,(T//6+1)*3))
ax = ax.flatten()    
    
np.random.seed(42)
s = np.copy(pattern)
corrupt = np.random.rand(M)>0.7 # weight
s[corrupt] = pattern2[corrupt]
print(np.sum((s-pattern)**2)/len(pattern))
ax[0].imshow(s.reshape(image.shape),cmap=plt.cm.gray)

for t in range(T):
    s = (0.5 + 0.5 * np.sign((w @ s) - bias2)).astype(int)
    ax[t+1].imshow(s.reshape(image.shape),cmap=plt.cm.gray)
    print(np.sum((s-pattern)**2)/len(pattern))
    plt.suptitle("Images of bias = 0.9 and weight = 0.7",fontsize=16)
    
# plotting bias = 0.5 and weight 0.5
fig, ax = plt.subplots(T//6+1,6, figsize=(16,(T//6+1)*3))
ax = ax.flatten()    
    
np.random.seed(42)
s = np.copy(pattern)
corrupt = np.random.rand(M)>0.5 # weight
s[corrupt] = pattern2[corrupt]
print(np.sum((s-pattern)**2)/len(pattern))
ax[0].imshow(s.reshape(image.shape),cmap=plt.cm.gray)

for t in range(T):
    s = (0.5 + 0.5 * np.sign((w @ s) - bias3)).astype(int)
    ax[t+1].imshow(s.reshape(image.shape),cmap=plt.cm.gray)
    print(np.sum((s-pattern)**2)/len(pattern))
    plt.suptitle("Images of bias = 0.5 and weight = 0.5",fontsize=16)
    
# plotting bias = 0.9 and weight 0.3
fig, ax = plt.subplots(T//6+1,6, figsize=(16,(T//6+1)*3))
ax = ax.flatten()    
    
np.random.seed(42)
s = np.copy(pattern)
corrupt = np.random.rand(M)>0.3 # weight
s[corrupt] = pattern2[corrupt]
print(np.sum((s-pattern)**2)/len(pattern))
ax[0].imshow(s.reshape(image.shape),cmap=plt.cm.gray)

for t in range(T):
    s = (0.5 + 0.5 * np.sign((w @ s) - bias2)).astype(int)
    ax[t+1].imshow(s.reshape(image.shape),cmap=plt.cm.gray)
    print(np.sum((s-pattern)**2)/len(pattern))
    plt.suptitle("Images of bias = 0.9 and weight = 0.3",fontsize=16)
    
'''
Exercise 3d)
There is good evidence that this model recapitulates at least some aspects of how memories are stored in the brain. Examine which aspects of this model are biologically plausible, and which are not.

Aspects of this model are biologically plausible

1) This model follows unsupervised learning, which is similar to the brain's system where it infers and finds correlations in learning without predefined outcomes.
2) This model is observed to function well as associative memory, even when the data is damaged. This shows a similarity to the brain's ability to recover memories and maintain functionality even after sustaining damage.
3) In this model, the input is distributed and stored across the entire network. As this model learns the correction weights for two patterns and stores them distributedly in the network, it resembles the way memories are distributed and stored across multiple neurons in the brain's neural network.
4) This model uses synchronous updates, where all neurons simultaneously calculate their activations and update their states. Also, it uses asynchronous updates which allow each neuron to compute its activation and update its state individually, one at a time, leading to gradual changes in the network's overall state. This reflects the behaviour of actual neurons in the brain, which can activate independently and change their states at different times.
5) Sparseness is used to control the activation patterns of the neural network. In the brain, not all neurons activate simultaneously, only specific neurons activate to process the necessary information, exhibiting a sparse activity pattern. This sparsity increases the efficiency of information processing and enhances the representation of important information.
Aspects of this model are NOT biologically plausible

1) In this model, the state of the neural network is binary, represented as 0 and 1. However, neurons in the actual brain process information using continuous analog signals rather than binary ones. Therefore, the binary approach of the model does not fully reflect the actual activity of neurons.
2) Since this model is based on the Hopfield model, the weights between neurons are symmetric. However, in the actual brain, the weights between neurons may not be symmetric.
3) In this model, the weights are determined randomly, but in the actual Hopfield model, weights are determined based on the Hebbian rule. The Hebbian rule is that the connection strength between neurons increases when their activities are positively correlated. This concept aligns with Connectionism, as proposed by Hebb, and it reflects a mechanism consistent with the actual activity of our neurons.
4) In this network model, it is assumed that all neurons are fully connected to each other, which may differ from the complex network structure of the actual brain. Neurons in the brain are not all interconnected but are linked in specific patterns.
'''
# these are the libraries we are going to use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from RM1985 import *
matplotlib.style.use('seaborn-v0_8-notebook')
import warnings
warnings.filterwarnings('ignore')
np.random.seed(2222)


#Encoding the input
word_to_wickelphone('kAm')
activate_word('kAm')

# Exercise 1a)
# Let's translate the ten high frequency words used in the paper:
# come/came, look/looked, feel/felt, have/had, make/made, get/got, give/gave, take/took, go/went, like/liked
high_frequency_verbs = ['come', 'look', 'feel', 'have', 'make', 'get', 'give', 'take', 'go', 'like']
base_high_frequency_verbs = ['k*m', 'luk', 'fEl', 'hav', 'mAk', 'get', 'giv', 'tAk', 'gO', 'lIk' ]
past_high_frequency_verbs = ['kAm', 'lukt', 'felt', 'had','mAd', 'got', 'gAv', 'tuk','went', 'likt' ]

#Exercise 1b)
#Store the shape of the wickelfeature represetation for came and the number of wickelfeatures activated (i.e., set to 1).
came_activation = np.array(activate_word('kAm'))#(460,)
came_shape = np.shape(came_activation) 
came_number_active_wickelfeatures = np.count_nonzero(came_activation) #16

#The Model
#Exercise 2a)
#Translate this function into code.
#1/1+𝑒^(−(𝑛𝑒𝑡−𝜃)/𝑇)
def rm_activation_function(net, theta, T=1.0):
    probability = 1/(1 + np.exp(-(net - theta)/T))
    return probability


#Exercise 2b)
#Let's set theta = 0.0 and plot the probability of firing as a function of the the weighted activation 
# net at T=1.0, T=0.5 and T=2.0 in one figure. We should consider weighted activation values between -5 and 5.
net_activation = np.arange(-5, 6)
p_T1 = rm_activation_function(net_activation, theta=0.0, T=1.0)
p_T05 = rm_activation_function(net_activation, theta=0.0, T=0.5)
p_T2 = rm_activation_function(net_activation, theta=0.0, T=2.0)

#plot RM's Perceptron Model
plt.figure()
plt.title("RM's Perceptron")
plt.plot(net_activation, p_T1, label='T=1.0', color='red')
plt.plot(net_activation, p_T05, label='T=0.5', color='blue')
plt.plot(net_activation, p_T2, label='T=2.0', color='green')
plt.xlabel("Weighted Activation")
plt.ylabel("Probability of Firing")
plt.xlim(-5,5)
plt.ylim(0,1)
plt.legend()
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


#U Shaped Curves
verbs = []
with open('c:\gitLocal\Cognitive science assignments\verbs.csv') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            print(line.strip('\n').split(','))
        else:
            verbs.append(line.strip('\n').split(','))
            # print(verbs[-1])

# add in the high frequency verbs you translated for us earlier
for i, word in enumerate(high_frequency_verbs):
    if word in ['look', 'like']:
        verbs.append([word, 'Regular', base_high_frequency_verbs[i], past_high_frequency_verbs[i], 'H'])
    else:
        verbs.append([word, 'Irregular', base_high_frequency_verbs[i], past_high_frequency_verbs[i], 'H'])
    #print(verbs[-1])

#First Stage: High Frequency Verbs    

base_wickel_HF = np.array([activate_word(w) for w in base_high_frequency_verbs]).T #.T: Transpose
past_wickel_HF = np.array([activate_word(w) for w in past_high_frequency_verbs]).T

percept = Perceptron(active=rm_activation_function)

percept.learn(base_wickel_HF, past_wickel_HF)
#array([0.97173913, 0.94782609, 0.93695652, 0.97173913, 0.95652174,
#       0.96956522, 0.95      , 0.96304348, 0.84347826, 0.93043478])

percept.score(base_wickel_HF, past_wickel_HF)
#array([0.9673913 , 0.95434783, 0.94347826, 0.9673913 , 0.95869565,
#       0.97391304, 0.9673913 , 0.96304348, 0.87173913, 0.92826087])


#Exercise 3a)
#Now let's divide the corpus verbs into two lists: 
# one for regular verbs and one for irregular verbs.
regular_verbs = []
irregular_verbs = []
for verb in verbs:
    if verb[1] == "Regular":
        regular_verbs.append(verb)
    elif verb[1] == "Irregular":
        irregular_verbs.append(verb)

#Exercise 3b)
#First convert the phonemes for base and past tense into wickelfeatures. 
# Then calculate the mean score of the model on irregular and regular verbs.
base_wickel_irregular = np.array([activate_word(w[2]) for w in irregular_verbs]).T
past_wickel_irregular = np.array([activate_word(w[3]) for w in irregular_verbs]).T
base_wickel_regular = np.array([activate_word(w[2]) for w in regular_verbs]).T
past_wickel_regular = np.array([activate_word(w[3]) for w in regular_verbs]).T

irregular_score = np.mean(percept.score(base_wickel_irregular, past_wickel_irregular))
regular_score = np.mean(percept.score(base_wickel_regular, past_wickel_regular))

# Let's initialize a new perceptron with our custom activation function
percept = Perceptron(active=rm_activation_function)

# Now let's loop through each data point to train and score
scores_regular = []
scores_irregular = []
for i in range(len(high_frequency_verbs)):
    percept.learn(base_wickel_HF[:,i, np.newaxis], past_wickel_HF[:,i,np.newaxis])
    scores_regular.append(np.mean(percept.score(base_wickel_regular, past_wickel_regular)))
    scores_irregular.append(np.mean(percept.score(base_wickel_irregular, past_wickel_irregular)))
    print(scores_regular[-1], scores_irregular[-1])

#Second Stage: Medium Frequency Verbs

#Exercise 4a)
#First, we need to extract the medium frequency verbs from the corpus verbs
base_med_frequency_verbs = []
past_med_frequency_verbs = []
for verb in verbs:
    if verb[4] == 'M':
        base_med_frequency_verbs.append(verb)
        past_med_frequency_verbs.append(verb)

#Exercise 4b)
#Second, we need to convert those those verbs into wickelfeatures.
base_wickel_MF = np.array([activate_word(w[2]) for w in base_med_frequency_verbs]).T
past_wickel_MF = np.array([activate_word(w[3]) for w in past_med_frequency_verbs]).T

#Exercise 4c)
#Calculate and store the scores for regular and irregular verbs in the variables scores_irregular_md and scores_regular_md.
scores_regular_md = []
scores_irregular_md = []
for i in range(len(base_med_frequency_verbs)): 
    percept.learn(base_wickel_MF[:,i, np.newaxis], past_wickel_MF[:,i,np.newaxis])
    scores_regular_md.append(np.mean(percept.score(base_wickel_regular, past_wickel_regular)))
    scores_irregular_md.append(np.mean(percept.score(base_wickel_irregular, past_wickel_irregular)))
    print(scores_regular_md[-1], scores_irregular_md[-1])

#Third Stage: Low Frequency Verbs

#Exercise 4d)
#First, we need to extract the low frequency verbs from the corpus verbs.
base_low_frequency_verbs = []
past_low_frequency_verbs = []

for verb in verbs:
    if verb[4] == 'L':
        base_low_frequency_verbs.append(verb)
        past_low_frequency_verbs.append(verb)
        
#Exercise 4e)
#Second, we need to convert those those verbs into wickelfeatures.
base_wickel_LF = np.array([activate_word(w[2]) for w in base_low_frequency_verbs]).T
past_wickel_LF = np.array([activate_word(w[3]) for w in past_low_frequency_verbs]).T

#Exercise 4f)
#Calculate and store the scores for regular and irregular verbs in the variables scores_irregular_low and scores_regular_low.
scores_regular_low = []
scores_irregular_low = []
for i in range(len(base_low_frequency_verbs)): 
    percept.learn(base_wickel_LF[:,i, np.newaxis], past_wickel_LF[:,i,np.newaxis])
    scores_regular_low.append(np.mean(percept.score(base_wickel_regular, past_wickel_regular)))
    scores_irregular_low.append(np.mean(percept.score(base_wickel_irregular, past_wickel_irregular)))
    print(scores_regular_low[-1], scores_irregular_low[-1])

#Exercise 5a)
#Let's see how well we replicated the U-Shape pattern by plotting the simulation output.

data_points = list(range(0,192)) 
total_scores_regular = scores_regular + scores_regular_md + scores_regular_low
total_scores_irregular = scores_irregular + scores_irregular_md + scores_irregular_low

plt.figure()
plt.plot(data_points, total_scores_regular, "o-", label="Regular Verbs", color="red", markersize=2, lw=1)
plt.plot(data_points, total_scores_irregular, "o-", label='Irregular Verbs', color="blue", markersize=2, lw=1)

plt.axvline(x=10, color="gray", linestyle="--")
plt.axvline(x=160, color="gray", linestyle="--")

plt.xlabel("Trials", fontsize=14)
plt.ylabel("Correct Wickelfeatures Ratio", fontsize=14)
plt.title("RM's Model of the Past Tense", fontsize=18)

plt.ylim(0.5, 1)
plt.xlim(0, 192)
plt.legend(loc="best")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

#Exercise 6
# Error Analysis
# Ordered according to frequency, based on Table 8, page 23 of the RM source paper.
# No distinction was made based on frequency.

#Type 1: No change: Verbs that do not change at all to form the past tense.
#Total 9 words
type1_verbs = ['beat', 'fit', 'set', 'spread', 'hit', 'cut', 'put', 'thrust', 'bid']
type1_bases = ['bEt', 'fit', 'set', 'spred', 'hit', 'kut', 'p*t', 'Trust', 'bid']
type1_pasts = ['bEt', 'fit', 'set', 'spred', 'hit', 'kut', 'p*t', 'Trust', 'bid']

#Type 2: /d/ to /t/ Change: Verbs that change a final /d/ to /t/ to form the past tense.
#Total 5 words
type2_verbs = ['build', 'send', 'spend', 'bend', 'lend']
type2_bases = ['bild', 'send', 'spend', 'bend', 'lend']
type2_pasts = ['bilt', 'sent', 'spent', 'bent', 'lent']

#Type 3: Internal Vowel Change + /t/ or /d/ Addition: Verbs that undergo an internal vowel change and also add a final /t/ or /d/.
#Total 16 words
type3_verbs = ['feel', 'deal', 'do', 'flee', 'tell', 'sell', 'hear', 'keep', 'leave', 'sleep', 'lose', 'mean', 'say', 'sweep', 'creep', 'weep']
type3_bases = ['fEl', 'dEl', 'dU', 'flE', 'tel', 'sel', 'hEr', 'kEp', 'lEv', 'slEp', 'lUz', 'mEn', 'sA', 'swEp', 'krEp', 'wEp']
type3_pasts = ['felt', 'delt', 'did', 'fled', 'tOld', 'sOld', 'herd', 'kept', 'left', 'slept', 'lost', 'ment', 'sed', 'swept', 'krept', 'wept']

#Type 4: Internal Vowel Change + Final Consonant Deletion + /t/ or /d/ Addition: Verbs that undergo an internal vowel change, delete a final consonant, and then add a final /t/ or /d/.
#Total 8 words
type4_verbs = ['have', 'make', 'think', 'buy', 'bring', 'seek', 'teach', 'catch']
type4_bases = ['hav', 'mAk', 'TiNk', 'bI', 'briN', 'sEk', 'tEC', 'kaC']
type4_pasts = ['had', 'mAd', 'Tot', 'bot', 'brot', 'sot', 'tot', 'kot']

#Type 5: Internal Vowel Change + Stem Ends in a Dental: Verbs that undergo an internal vowel change whose stems end in a dental.
#Total 22 words
type5_verbs = ['get', 'meet', 'shoot', 'write', 'lead', 'understand', 'sit', 'mislead', 'bleed', 'feed', 'stand', 'light', 'find', 'fight', 'read', 'meet', 'hide', 'hold', 'ride', 'breed', 'wind', 'grind']
type5_bases = ['get', 'mEt', 'Sut', 'rIt', 'lEd', '*nderstand', 'sit', 'mislEd', 'blEd', 'fEd', 'stand', 'lIt', 'fInd', 'fIt', 'rEd', 'mEt', 'hId', 'hOld', 'rId', 'brEd', 'wInd', 'grInd']
type5_pasts = ['got', 'met', 'Sot', 'rOt', 'led', '*nderstUd', 'sat', 'misled', 'bled', 'fed', 'stUd', 'lit', 'fWnd', 'fot', 'red', 'met', 'hid', 'held', 'rOd', 'bred', 'wWnd', 'grWnd']

#Type 6a: Vowel Change /i/ to /a/: Verbs that undergo a vowel change from /i/ to /a/. 
#Total 4 words
type6a_verbs = ['drink', 'ring', 'sing', 'swim']
type6a_bases = ['driNk', 'riN', 'siN', 'swim']
type6a_pasts = ['draNk', 'raN', 'saN', 'swam']

#Type 6b: Internal Vowel Change /i/ or /a/ to /u/: Verbs that undergo an internal vowel change from /i/ or /a/ to /u/.
#Total 6 words
type6b_verbs = ['drag', 'hang', 'swing', 'dig', 'cling', 'stick']
type6b_bases =  ['jrag', 'haN', 'swiN', 'dig', 'kliN', 'stik']
type6b_pasts = ['jr*g', 'h*N', 'swaN', 'd*g', 'klaN', 'st*k']

#Type 7: Other Internal Vowel Changes: Verbs that undergo other forms of internal vowel changes.
#Total 18 words
type7_verbs = ['give', 'take', 'come', 'shake', 'arise', 'rise', 'run', 'become', 'bear', 'wear', 'speak', 'brake', 'drive', 'strike', 'fall', 'freeze', 'choose', 'tear']
type7_bases = ['giv', 'tAk', 'k*m', 'Sak', '*rIz', 'rIz', 'r*n', 'bEk*m', 'bAr', 'wAr', 'spEk', 'brAk', 'jrIv', 'strIk', 'fol', 'frEz', 'Cuz', 'tAr']
type7_pasts = ['gAv', 'tuk', 'kAm', 'Suk', '*rOz', 'rOz', 'ran', 'bEkAm', 'bor', 'wor', 'spOk', 'brOk', 'jrOv', 'str*k', 'fel', 'frOz', 'COz', 'tor']

#Type 8:Vowel Change + Ends in a Diphthong: Verbs that undergo a vowel change and end in a diphthongal sequence.
#Total 8 words
type8_verbs = ['go', 'throw', 'blow', 'grow', 'draw', 'fly', 'know', 'see']
type8_bases = ['gO', 'DrO', 'blO', 'grO', 'jro', 'flI', 'nO', 'sE']
type8_pasts = ['went', 'DrU', 'blU', 'grU', 'jrU', 'flU', 'nU', 'so']

#Converting the phonemes into wickelfeatures.

#Type 1 
base_wickel_type1 = np.array([activate_word(w) for w in type1_bases]).T
past_wickel_type1 = np.array([activate_word(w) for w in type1_pasts]).T

#Type 2
base_wickel_type2 = np.array([activate_word(w) for w in type2_bases]).T
past_wickel_type2 = np.array([activate_word(w) for w in type2_pasts]).T

#Type 3
base_wickel_type3 = np.array([activate_word(w) for w in type3_bases]).T
past_wickel_type3 = np.array([activate_word(w) for w in type3_pasts]).T

#Type 4
base_wickel_type4 = np.array([activate_word(w) for w in type4_bases]).T
past_wickel_type4 = np.array([activate_word(w) for w in type4_pasts]).T

#Type 5
base_wickel_type5 = np.array([activate_word(w) for w in type5_bases]).T
past_wickel_type5 = np.array([activate_word(w) for w in type5_pasts]).T

#Type 6a
base_wickel_type6a = np.array([activate_word(w) for w in type6a_bases]).T
past_wickel_type6a = np.array([activate_word(w) for w in type6a_pasts]).T

#Type 6b
base_wickel_type6b = np.array([activate_word(w) for w in type6b_bases]).T
past_wickel_type6b = np.array([activate_word(w) for w in type6b_pasts]).T

#Type 7
base_wickel_type7 = np.array([activate_word(w) for w in type7_bases]).T
past_wickel_type7 = np.array([activate_word(w) for w in type7_pasts]).T

#Type 8
base_wickel_type8 = np.array([activate_word(w) for w in type8_bases]).T
past_wickel_type8 = np.array([activate_word(w) for w in type8_pasts]).T


'''
Justifying how to conduct modelling
The two models have different underlying perspectives and approaches in acquiring language knowledge.
The first model was based on frequency, aligning with empirical thinking. 
In contrast, the second model was based on the type of words, which corresponds to rational thinking. 
If we apply this to language acquisition in everyday life for infants, the first modelling is continuous 
because it is based on experiences in a random environment, measured by "frequency," literally a measure of occurrence probability. 
Therefore, the first modeling is suitable for learning within one model from high frequency to low frequency. 
However, the second modelling is independent because it was distinguished based on standardised definitions or rules, 
meaning prior knowledge or experience has little or no impact on subsequent learning. 
Thus, it was determined that the second modeling is appropriate for training a new instance model for each type.

'''

#Learn and score the new model

#Reset the model to make a new instance of the RM perceptron whenever the type changes.
#new_type x = Training a new model for each type.

#Type 1
new_percept1 = Perceptron(active=rm_activation_function) #Type 1 model

new_type1_score =[]
for i in range(len(type1_verbs)):
    new_percept1.learn(base_wickel_type1[:,i, np.newaxis], past_wickel_type1[:,i,np.newaxis])
    new_type1_score.append(np.mean(new_percept1.score(base_wickel_type1, past_wickel_type1)))
    print(new_type1_score[-1])

#Type 2
new_percept2 = Perceptron(active=rm_activation_function) #Type 2 model

new_type2_score =[]
for i in range(len(type2_verbs)):
    new_percept2.learn(base_wickel_type2[:,i, np.newaxis], past_wickel_type2[:,i,np.newaxis])
    new_type2_score.append(np.mean(new_percept2.score(base_wickel_type2, past_wickel_type2)))
    print(new_type2_score[-1])

#Type 3
new_percept3 = Perceptron(active=rm_activation_function) #Type 3 model

new_type3_score =[]
for i in range(len(type3_verbs)):
    new_percept3.learn(base_wickel_type3[:,i, np.newaxis], past_wickel_type3[:,i,np.newaxis])
    new_type3_score.append(np.mean(new_percept3.score(base_wickel_type3, past_wickel_type3)))
    print(new_type3_score[-1])

#Type 4
new_percept4 = Perceptron(active=rm_activation_function) #Type 4 model

new_type4_score =[]
for i in range(len(type4_verbs)):
    new_percept4.learn(base_wickel_type4[:,i, np.newaxis], past_wickel_type4[:,i,np.newaxis])
    new_type4_score.append(np.mean(new_percept4.score(base_wickel_type4, past_wickel_type4)))
    print(new_type4_score[-1])   

#Type 5
new_percept5 = Perceptron(active=rm_activation_function) #Type 5 model

new_type5_score =[]
for i in range(len(type5_verbs)):
    new_percept5.learn(base_wickel_type5[:,i, np.newaxis], past_wickel_type5[:,i,np.newaxis])
    new_type5_score.append(np.mean(new_percept5.score(base_wickel_type5, past_wickel_type5)))
    print(new_type5_score[-1])

#Type 6a
new_percept6a = Perceptron(active=rm_activation_function) #Type 6a model

new_type6a_score =[]
for i in range(len(type6a_verbs)):
    new_percept6a.learn(base_wickel_type6a[:,i, np.newaxis], past_wickel_type6a[:,i,np.newaxis])
    new_type6a_score.append(np.mean(new_percept6a.score(base_wickel_type6a, past_wickel_type6a)))
    print(new_type6a_score[-1])

#Type 6b
new_percept6b = Perceptron(active=rm_activation_function) #Type 6b model

new_type6b_score =[]
for i in range(len(type6b_verbs)):
    new_percept6b.learn(base_wickel_type6b[:,i, np.newaxis], past_wickel_type6b[:,i,np.newaxis])
    new_type6b_score.append(np.mean(new_percept6b.score(base_wickel_type6b, past_wickel_type6b)))
    print(new_type6b_score[-1])

#Type 7
new_percept7 = Perceptron(active=rm_activation_function) #Type 7 model

new_type7_score =[]
for i in range(len(type7_verbs)):
    new_percept7.learn(base_wickel_type7[:,i, np.newaxis], past_wickel_type7[:,i,np.newaxis])
    new_type7_score.append(np.mean(new_percept7.score(base_wickel_type7, past_wickel_type7)))
    print(new_type7_score[-1])

#Type 8
new_percept8 = Perceptron(active=rm_activation_function) #Type 8 model

new_type8_score =[]
for i in range(len(type8_verbs)):
    new_percept8.learn(base_wickel_type8[:,i, np.newaxis], past_wickel_type8[:,i,np.newaxis])
    new_type8_score.append(np.mean(new_percept8.score(base_wickel_type8, past_wickel_type8)))
    print(new_type8_score[-1])
 
    
#Plotting the data into two types of graphs

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

#Graph 1: Individual learning curves for each verb type.
new_colors = ["y", "g", "k", "m", "purple", "b", "r", "orange", "c"] #Distinguishing the type with different colours
new_types = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6a", "Type 6b", "Type 7", "Type 8"]
new_types_scores = [new_type1_score, new_type2_score, new_type3_score, 
                    new_type4_score, new_type5_score, new_type6a_score, 
                    new_type6b_score, new_type7_score, new_type8_score]

for i, scores in enumerate(new_types_scores):
    axes[0].plot(scores, label=new_types[i],color=new_colors[i], lw=1)

axes[0].set_xlabel("Trials", fontsize=14)
axes[0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[0].set_title("Individual Lines", fontsize=18)
axes[0].legend(loc="lower right")
axes[0].set_xlim(0, 25)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)


#Graph 2: A single-line graph showing the learning curves for each verb type, ordered from type 1 to type 8.
data_points = list(range(0,96)) 
new_total_types_scores = (new_type1_score + new_type2_score + new_type3_score + 
                      new_type4_score + new_type5_score + new_type6a_score + 
                      new_type6b_score + new_type7_score + new_type8_score)

axes[1].plot(data_points, new_total_types_scores, "o-", label="Regular Verbs", color="red", markersize=2, lw=1)         
axes[1].set_xlabel("Trials", fontsize=14)
axes[1].set_title("Single-Line", fontsize=18)
axes[1].set_ylim(0.5, 1)
axes[1].set_xlim(0, 96)
axes[1].legend(loc="lower right")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

#Demarcating the types
axes[1].axvline(x=9, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=14, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=30, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=38, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=60, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=64, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=70, color="gray", linestyle="--", lw=1)
axes[1].axvline(x=88, color="gray", linestyle="--", lw=1)


plt.suptitle("Learning Curves for Different Types of Irregular Verbs", fontsize=18)

plt.show()

'''
Hypothesis and Expectations
As the order from high to low frequency was determined from an empirical perspective, 
it is thought that the criteria and sequence of training the model is especially important from the rational perspective(the second model). 
Similar to how the first model prioritises training on frequently encounted words, 
it is considered beneficial for the second modeling approach to organise the learning sequence based on difficultiy of types rules. 

When referencing individually learned data, the rate of change corresponding to 
the slope of the graph and the starting point were thought as measures of rule difficulty. 
Therefore, it is believed that starting the training the verb type that have a high average rate of change and high starting scores, 
which correspond to easier level, could closely replicate the shape of the first model.
'''


'''
First criterion for the training: Average change
Since the modeling was done independently, it's possible to judge how complex the rules are for each type. 
Therefore, it was thought that this could be determined through the calculation of the average rate of change.
'''

#Calculating the average change of each verb type

for index, scores in enumerate(new_types_scores):
    sum = 0
    for i in range(1, len(scores)):
        # Calculate the change between consecutive scores.
        change = scores[i] - scores[i-1] 
        sum += change
    # Calculate the average change by dividing the sum of changes by the number of pairs of scores.
    average_change = sum / (len(scores) - 1)
    average_change_percent = average_change * 100
    
    #The order of type 1 to type 8 average change and its percentage
    print(f"{average_change}, {average_change_percent:.2f}%")

'''
Justifying the order of training
It was determined that a high rate of change means there is consistente regularity within the types, making them easier to acquire. 
This was expected to have a similar effect to training in the order of high frequency.
'''

#Learn and score the new model
#The original model for comparing the results of changes in the training. The sequence is trained in the order of types.

original_percept = Perceptron(active=rm_activation_function)

original_type1_score =[]
for i in range(len(type1_verbs)):
    original_percept.learn(base_wickel_type1[:,i, np.newaxis], past_wickel_type1[:,i,np.newaxis])
    original_type1_score.append(np.mean(original_percept.score(base_wickel_type1, past_wickel_type1)))
    print(original_type1_score[-1])

#Type 2
original_type2_score =[]
for i in range(len(type2_verbs)):
    original_percept.learn(base_wickel_type2[:,i, np.newaxis], past_wickel_type2[:,i,np.newaxis])
    original_type2_score.append(np.mean(original_percept.score(base_wickel_type2, past_wickel_type2)))
    print(original_type2_score[-1])

#Type 3
original_type3_score =[]
for i in range(len(type3_verbs)):
    original_percept.learn(base_wickel_type3[:,i, np.newaxis], past_wickel_type3[:,i,np.newaxis])
    original_type3_score.append(np.mean(original_percept.score(base_wickel_type3, past_wickel_type3)))
    print(original_type3_score[-1])

#Type 4
original_type4_score =[]
for i in range(len(type4_verbs)):
    original_percept.learn(base_wickel_type4[:,i, np.newaxis], past_wickel_type4[:,i,np.newaxis])
    original_type4_score.append(np.mean(original_percept.score(base_wickel_type4, past_wickel_type4)))
    print(original_type4_score[-1])   

#Type 5
original_type5_score =[]
for i in range(len(type5_verbs)):
    original_percept.learn(base_wickel_type5[:,i, np.newaxis], past_wickel_type5[:,i,np.newaxis])
    original_type5_score.append(np.mean(original_percept.score(base_wickel_type5, past_wickel_type5)))
    print(original_type5_score[-1])

#Type 6a
original_type6a_score =[]
for i in range(len(type6a_verbs)):
    original_percept.learn(base_wickel_type6a[:,i, np.newaxis], past_wickel_type6a[:,i,np.newaxis])
    original_type6a_score.append(np.mean(original_percept.score(base_wickel_type6a, past_wickel_type6a)))
    print(original_type6a_score[-1])

#Type 6b
original_type6b_score =[]
for i in range(len(type6b_verbs)):
    original_percept.learn(base_wickel_type6b[:,i, np.newaxis], past_wickel_type6b[:,i,np.newaxis])
    original_type6b_score.append(np.mean(original_percept.score(base_wickel_type6b, past_wickel_type6b)))
    print(original_type6b_score[-1])

#Type 7
original_type7_score =[]
for i in range(len(type7_verbs)):
    original_percept.learn(base_wickel_type7[:,i, np.newaxis], past_wickel_type7[:,i,np.newaxis])
    original_type7_score.append(np.mean(original_percept.score(base_wickel_type7, past_wickel_type7)))
    print(original_type7_score[-1])

#Type 8
original_type8_score =[]
for i in range(len(type8_verbs)):
    original_percept.learn(base_wickel_type8[:,i, np.newaxis], past_wickel_type8[:,i,np.newaxis])
    original_type8_score.append(np.mean(original_percept.score(base_wickel_type8, past_wickel_type8)))
    print(original_type8_score[-1])
    

#A model trained in the order of the highest avaerage rate of change. 

rearrange_high_new_percept = Perceptron(active=rm_activation_function)

rearrange_high_type6a_scores =[]
for i in range(len(type6a_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type6a[:,i, np.newaxis], past_wickel_type6a[:,i,np.newaxis])
    rearrange_high_type6a_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type6a, past_wickel_type6a)))
    print(rearrange_high_type6a_scores[-1])
    
rearrange_high_type2_scores =[]
for i in range(len(type2_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type2[:,i, np.newaxis], past_wickel_type2[:,i,np.newaxis])
    rearrange_high_type2_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type2, past_wickel_type2)))
    print(rearrange_high_type2_scores[-1])
    
rearrange_high_type6b_scores =[]
for i in range(len(type6b_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type6b[:,i, np.newaxis], past_wickel_type6b[:,i,np.newaxis])
    rearrange_high_type6b_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type6b, past_wickel_type6b)))
    print(rearrange_high_type6b_scores[-1])

rearrange_high_type8_scores =[]    
for i in range(len(type8_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type8[:,i, np.newaxis], past_wickel_type8[:,i,np.newaxis])
    rearrange_high_type8_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type8, past_wickel_type8)))
    print(rearrange_high_type8_scores[-1])
    
rearrange_high_type4_scores =[]
for i in range(len(type4_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type4[:,i, np.newaxis], past_wickel_type4[:,i,np.newaxis])
    rearrange_high_type4_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type4, past_wickel_type4)))
    print(rearrange_high_type4_scores[-1])   
    
rearrange_high_type1_scores =[]
for i in range(len(type1_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type1[:,i, np.newaxis], past_wickel_type1[:,i,np.newaxis])
    rearrange_high_type1_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type1, past_wickel_type1)))
    print(rearrange_high_type1_scores[-1])
    
rearrange_high_type3_scores =[]
for i in range(len(type3_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type3[:,i, np.newaxis], past_wickel_type3[:,i,np.newaxis])
    rearrange_high_type3_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type3, past_wickel_type3)))
    print(rearrange_high_type3_scores[-1])

rearrange_high_type7_scores =[]
for i in range(len(type7_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type7[:,i, np.newaxis], past_wickel_type7[:,i,np.newaxis])
    rearrange_high_type7_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type7, past_wickel_type7)))
    print(rearrange_high_type7_scores[-1])

rearrange_high_type5_scores =[]
for i in range(len(type5_verbs)):
    rearrange_high_new_percept.learn(base_wickel_type5[:,i, np.newaxis], past_wickel_type5[:,i,np.newaxis])
    rearrange_high_type5_scores.append(np.mean(rearrange_high_new_percept.score(base_wickel_type5, past_wickel_type5)))
    print(rearrange_high_type5_scores[-1])
    
    
#A model trained in the order of the highest avaerage rate of change. 
# A model for the comparison with "original_percept" and "rearrange_high_new_percept". 

rearrange_low_new_percept = Perceptron(active=rm_activation_function)

rearrange_low_type5_scores =[]
for i in range(len(type5_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type5[:,i, np.newaxis], past_wickel_type5[:,i,np.newaxis])
    rearrange_low_type5_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type5, past_wickel_type5)))
    print(rearrange_low_type5_scores[-1])
    
rearrange_low_type7_scores =[]
for i in range(len(type7_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type7[:,i, np.newaxis], past_wickel_type7[:,i,np.newaxis])
    rearrange_low_type7_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type7, past_wickel_type7)))
    print(rearrange_low_type7_scores[-1])
    
rearrange_low_type3_scores =[]
for i in range(len(type3_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type3[:,i, np.newaxis], past_wickel_type3[:,i,np.newaxis])
    rearrange_low_type3_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type3, past_wickel_type3)))
    print(rearrange_low_type3_scores[-1])
    
rearrange_low_type1_scores =[]
for i in range(len(type1_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type1[:,i, np.newaxis], past_wickel_type1[:,i,np.newaxis])
    rearrange_low_type1_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type1, past_wickel_type1)))
    print(rearrange_low_type1_scores[-1])
    
rearrange_low_type4_scores =[]
for i in range(len(type4_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type4[:,i, np.newaxis], past_wickel_type4[:,i,np.newaxis])
    rearrange_low_type4_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type4, past_wickel_type4)))
    print(rearrange_low_type4_scores[-1])  
    
rearrange_low_type8_scores =[]    
for i in range(len(type8_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type8[:,i, np.newaxis], past_wickel_type8[:,i,np.newaxis])
    rearrange_low_type8_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type8, past_wickel_type8)))
    print(rearrange_low_type8_scores[-1])
    
rearrange_low_type6b_scores =[]
for i in range(len(type6b_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type6b[:,i, np.newaxis], past_wickel_type6b[:,i,np.newaxis])
    rearrange_low_type6b_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type6b, past_wickel_type6b)))
    print(rearrange_low_type6b_scores[-1])
    
rearrange_low_type2_scores =[]
for i in range(len(type2_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type2[:,i, np.newaxis], past_wickel_type2[:,i,np.newaxis])
    rearrange_low_type2_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type2, past_wickel_type2)))
    print(rearrange_low_type2_scores[-1])    

rearrange_low_type6a_scores =[]
for i in range(len(type6a_verbs)):
    rearrange_low_new_percept.learn(base_wickel_type6a[:,i, np.newaxis], past_wickel_type6a[:,i,np.newaxis])
    rearrange_low_type6a_scores.append(np.mean(rearrange_low_new_percept.score(base_wickel_type6a, past_wickel_type6a)))
    print(rearrange_low_type6a_scores[-1])
    
    
#Plotting the original and change average rearranged data into two types of graphs

fig, axes = plt.subplots(3, 2, figsize=(18, 16), sharey=True)

#Graph 1: Individual learning curves for "original_percept".
original_colors = ["y", "g", "k", "m", "purple", "b", "r", "orange", "c"]
original_types = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6a", "Type 6b", "Type 7", "Type 8"]
original_types_scores = [original_type1_score, original_type2_score, original_type3_score, 
                         original_type4_score, original_type5_score, original_type6a_score, 
                         original_type6b_score, original_type7_score, original_type8_score]

for i, scores in enumerate(original_types_scores):
    axes[0,0].plot(scores, label=original_types[i], color=original_colors[i], lw=1)

axes[0,0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[0,0].set_title("Original Individual Lines", fontsize=18)
axes[0,0].set_xlim(0, 25)
axes[0,0].legend(loc= "lower right")
axes[0,0].spines["top"].set_visible(False)
axes[0,0].spines["right"].set_visible(False)


#Graph 2: A single line graph of learning curve for "original_percept".
data_points = list(range(0,96)) 
original_total_types_scores = (original_type1_score + original_type2_score + original_type3_score +
                               original_type4_score + original_type5_score + original_type6a_score + 
                               original_type6b_score + original_type7_score + original_type8_score)

axes[0,1].plot(data_points, original_total_types_scores, "o-", label="Original Verbs", color="red", markersize=2, lw=1)
axes[0,1].set_title("Original Single-Line", fontsize=18)
axes[0,1].set_ylim(0.5, 1)
axes[0,1].set_xlim(0, 96)
axes[0,1].legend(loc= "lower right")
axes[0,1].spines["top"].set_visible(False)
axes[0,1].spines["right"].set_visible(False)

#Demarcating the types
axes[0,1].axvline(x=9, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=14, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=30, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=38, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=60, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=64, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=70, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=88, color="gray", linestyle="--", lw=1)


#Graph 3: Individual learning curves for "rearrange_high_new_percept".
rearrange_high_colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple"]
rearrange_high_types = ["Type 6a", "Type 2", "Type 6b", "Type 8", "Type 4", "Type 1", "Type 3", "Type 7", "Type 5"]
rearrange_high_types_scores = [rearrange_high_type6a_scores, rearrange_high_type2_scores, rearrange_high_type6b_scores, 
                               rearrange_high_type8_scores, rearrange_high_type4_scores, rearrange_high_type1_scores, 
                               rearrange_high_type3_scores, rearrange_high_type7_scores, rearrange_high_type5_scores]

for i, scores in enumerate(rearrange_high_types_scores):
    axes[1,0].plot(scores, label=rearrange_high_types[i], color=rearrange_high_colors[i], lw=1)

axes[1,0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[1,0].set_title("Rearranged High Individual Lines", fontsize=18)
axes[1,0].set_xlim(0, 25)
axes[1,0].legend(loc= "lower right")
axes[1,0].spines["top"].set_visible(False)
axes[1,0].spines["right"].set_visible(False)


#Graph 4: A single-line graph showing the learning curves for "rearrange_high_new_percept".
data_points = list(range(0,96)) 
rearrange_high_total_types_scores = (rearrange_high_type6a_scores + rearrange_high_type2_scores + rearrange_high_type6b_scores + 
                                    rearrange_high_type8_scores + rearrange_high_type4_scores + rearrange_high_type1_scores + 
                                    rearrange_high_type3_scores + rearrange_high_type7_scores + rearrange_high_type5_scores)

axes[1,1].plot(data_points, rearrange_high_total_types_scores, "o-", label="Rearranged High Verbs", color="red", markersize=2, lw=1)
axes[1,1].set_title("Rearranged High Single-Line", fontsize=18)
axes[1,1].set_ylim(0.5, 1)
axes[1,1].set_xlim(0, 96)
axes[1,1].legend(loc= "lower right")
axes[1,1].spines['top'].set_visible(False)
axes[1,1].spines['right'].set_visible(False)

#Demarcating the types
axes[1,1].axvline(x=4, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=9, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=15, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=23, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=31, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=40, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=56, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=74, color="gray", linestyle="--", lw=1)


#Graph 5: Individual learning curves for "rearrange_low_new_percept".
rearrange_low_colors = ['purple', 'orange', 'k', 'y', 'm', 'c', 'r', 'g', 'b']
rearrange_low_types = ['Type 5', 'Type 7', 'Type 3', 'Type 1', 'Type 4', 'Type 8', 'Type 6b', 'Type 2', 'Type 6a']
rearrange_low_types_scores = [rearrange_low_type5_scores, rearrange_low_type7_scores, rearrange_low_type3_scores,
                              rearrange_low_type1_scores, rearrange_low_type4_scores, rearrange_low_type8_scores,
                              rearrange_low_type6b_scores, rearrange_low_type2_scores, rearrange_low_type6a_scores]
for i, scores in enumerate(rearrange_low_types_scores):
    axes[2,0].plot(scores, label=rearrange_low_types[i], color=rearrange_low_colors[i], lw=1)

axes[2,0].set_xlabel("Trials", fontsize=14)
axes[2,0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[2,0].set_title("Rearranged Low Individual Lines", fontsize=18)
axes[2,0].set_xlim(0, 25)
axes[2,0].legend(loc= "lower right")
axes[2,0].spines["top"].set_visible(False)
axes[2,0].spines["right"].set_visible(False)


#Graph 6: A single-line graph showing the learning curves for "rearrange_low_new_percept".
data_points = list(range(0,96)) 
rearrange_low_total_types_scores = (rearrange_low_type5_scores + rearrange_low_type7_scores + rearrange_low_type3_scores + 
                                    rearrange_low_type1_scores + rearrange_low_type4_scores + rearrange_low_type8_scores + 
                                    rearrange_low_type6b_scores + rearrange_low_type2_scores + rearrange_low_type6a_scores)

axes[2,1].plot(data_points, rearrange_low_total_types_scores, "o-", label="Rearranged Low Verbs", color="red", markersize=2, lw=1)
axes[2,1].set_xlabel("Trials", fontsize=14)
axes[2,1].set_title("Rearranged Low Single-Line", fontsize=18)
axes[2,1].set_ylim(0.5, 1)
axes[2,1].set_xlim(0, 96)
axes[2,1].legend(loc= "lower right")
axes[2,1].spines['top'].set_visible(False)
axes[2,1].spines['right'].set_visible(False)

#Demarcating the types
axes[2,1].axvline(x=22, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=40, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=56, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=65, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=73, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=81, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=87, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=92, color="gray", linestyle="--", lw=1)


plt.suptitle('Learning Curves for Original Verbs VS Change Average Rearranged Verbs', fontsize=18)

plt.show()

'''
Second criterion for the training: Starting score
It was determined that a high starting score means the regularity is relatively simple and easy to acquire. 
This was expected to have a similar effect to training in the order of high frequency.
'''


start_score = [new_type1_score[0], new_type2_score[0], new_type3_score[0],
               new_type4_score[0], new_type5_score[0], new_type6a_score[0],
               new_type6b_score[0], new_type7_score[0], new_type8_score[0]]

print(start_score)
 
   
#A model trained in the order of the highest starting score.

high_start_percept = Perceptron(active=rm_activation_function)

#Type 2
high_start_type2_score =[]
for i in range(len(type2_verbs)):
    high_start_percept.learn(base_wickel_type2[:,i, np.newaxis], past_wickel_type2[:,i,np.newaxis])
    high_start_type2_score.append(np.mean(high_start_percept.score(base_wickel_type2, past_wickel_type2)))
    print(high_start_type2_score[-1])

#Type 6a
high_start_type6a_score =[]
for i in range(len(type6a_verbs)):
    high_start_percept.learn(base_wickel_type6a[:,i, np.newaxis], past_wickel_type6a[:,i,np.newaxis])
    high_start_type6a_score.append(np.mean(high_start_percept.score(base_wickel_type6a, past_wickel_type6a)))
    print(high_start_type6a_score[-1])
    
#Type 6b
high_start_type6b_score =[]
for i in range(len(type6b_verbs)):
    high_start_percept.learn(base_wickel_type6b[:,i, np.newaxis], past_wickel_type6b[:,i,np.newaxis])
    high_start_type6b_score.append(np.mean(high_start_percept.score(base_wickel_type6b, past_wickel_type6b)))
    print(high_start_type6b_score[-1])
    
#Type 1   
high_start_type1_score =[]
for i in range(len(type1_verbs)):
    high_start_percept.learn(base_wickel_type1[:,i, np.newaxis], past_wickel_type1[:,i,np.newaxis])
    high_start_type1_score.append(np.mean(high_start_percept.score(base_wickel_type1, past_wickel_type1)))
    print(high_start_type1_score[-1]) 
    
#Type 7
high_start_type7_score =[]
for i in range(len(type7_verbs)):
    high_start_percept.learn(base_wickel_type7[:,i, np.newaxis], past_wickel_type7[:,i,np.newaxis])
    high_start_type7_score.append(np.mean(high_start_percept.score(base_wickel_type7, past_wickel_type7)))
    print(high_start_type7_score[-1])
    
#Type 3
high_start_type3_score =[]
for i in range(len(type3_verbs)):
    high_start_percept.learn(base_wickel_type3[:,i, np.newaxis], past_wickel_type3[:,i,np.newaxis])
    high_start_type3_score.append(np.mean(high_start_percept.score(base_wickel_type3, past_wickel_type3)))
    print(high_start_type3_score[-1])
    
#Type 5
high_start_type5_score =[]
for i in range(len(type5_verbs)):
    high_start_percept.learn(base_wickel_type5[:,i, np.newaxis], past_wickel_type5[:,i,np.newaxis])
    high_start_type5_score.append(np.mean(high_start_percept.score(base_wickel_type5, past_wickel_type5)))
    print(high_start_type5_score[-1])
    
#Type 4
high_start_type4_score =[]
for i in range(len(type4_verbs)):
    high_start_percept.learn(base_wickel_type4[:,i, np.newaxis], past_wickel_type4[:,i,np.newaxis])
    high_start_type4_score.append(np.mean(high_start_percept.score(base_wickel_type4, past_wickel_type4)))
    print(high_start_type4_score[-1]) 
    
#Type 8
high_start_type8_score =[]
for i in range(len(type8_verbs)):
    high_start_percept.learn(base_wickel_type8[:,i, np.newaxis], past_wickel_type8[:,i,np.newaxis])
    high_start_type8_score.append(np.mean(high_start_percept.score(base_wickel_type8, past_wickel_type8)))
    print(high_start_type8_score[-1])

#A model trained in the order of the lowest starting score.

low_start_percept = Perceptron(active=rm_activation_function)

#Type 8
low_start_type8_score =[]
for i in range(len(type8_verbs)):
    low_start_percept.learn(base_wickel_type8[:,i, np.newaxis], past_wickel_type8[:,i,np.newaxis])
    low_start_type8_score.append(np.mean(low_start_percept.score(base_wickel_type8, past_wickel_type8)))
    print(low_start_type8_score[-1])

#Type 4
low_start_type4_score =[]
for i in range(len(type4_verbs)):
    low_start_percept.learn(base_wickel_type4[:,i, np.newaxis], past_wickel_type4[:,i,np.newaxis])
    low_start_type4_score.append(np.mean(low_start_percept.score(base_wickel_type4, past_wickel_type4)))
    print(low_start_type4_score[-1])  

#Type 5
low_start_type5_score =[]
for i in range(len(type5_verbs)):
    low_start_percept.learn(base_wickel_type5[:,i, np.newaxis], past_wickel_type5[:,i,np.newaxis])
    low_start_type5_score.append(np.mean(low_start_percept.score(base_wickel_type5, past_wickel_type5)))
    print(low_start_type5_score[-1])
    
#Type 3
low_start_type3_score =[]
for i in range(len(type3_verbs)):
    low_start_percept.learn(base_wickel_type3[:,i, np.newaxis], past_wickel_type3[:,i,np.newaxis])
    low_start_type3_score.append(np.mean(low_start_percept.score(base_wickel_type3, past_wickel_type3)))
    print(low_start_type3_score[-1])
    
#Type 7
low_start_type7_score =[]
for i in range(len(type7_verbs)):
    low_start_percept.learn(base_wickel_type7[:,i, np.newaxis], past_wickel_type7[:,i,np.newaxis])
    low_start_type7_score.append(np.mean(low_start_percept.score(base_wickel_type7, past_wickel_type7)))
    print(low_start_type7_score[-1])

#Type 1   
low_start_type1_score =[]
for i in range(len(type1_verbs)):
    low_start_percept.learn(base_wickel_type1[:,i, np.newaxis], past_wickel_type1[:,i,np.newaxis])
    low_start_type1_score.append(np.mean(low_start_percept.score(base_wickel_type1, past_wickel_type1)))
    print(low_start_type1_score[-1]) 

#Type 6b
low_start_type6b_score =[]
for i in range(len(type6b_verbs)):
    low_start_percept.learn(base_wickel_type6b[:,i, np.newaxis], past_wickel_type6b[:,i,np.newaxis])
    low_start_type6b_score.append(np.mean(low_start_percept.score(base_wickel_type6b, past_wickel_type6b)))
    print(low_start_type6b_score[-1])

#Type 6a
low_start_type6a_score =[]
for i in range(len(type6a_verbs)):
    low_start_percept.learn(base_wickel_type6a[:,i, np.newaxis], past_wickel_type6a[:,i,np.newaxis])
    low_start_type6a_score.append(np.mean(low_start_percept.score(base_wickel_type6a, past_wickel_type6a)))
    print(low_start_type6a_score[-1])

#Type 2
low_start_type2_score =[]
for i in range(len(type2_verbs)):
    low_start_percept.learn(base_wickel_type2[:,i, np.newaxis], past_wickel_type2[:,i,np.newaxis])
    low_start_type2_score.append(np.mean(low_start_percept.score(base_wickel_type2, past_wickel_type2)))
    print(low_start_type2_score[-1])
    

#Plotting the original and starting score rearranged data into two types of graphs

fig, axes = plt.subplots(3, 2, figsize=(18, 16), sharey=True)

#Graph 1: Individual learning curves for "original_percept".
original_colors = ["y", "g", "k", "m", "purple", "b", "r", "orange", "c"]
original_types = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6a", "Type 6b", "Type 7", "Type 8"]
original_types_scores = [original_type1_score, original_type2_score, original_type3_score, 
                         original_type4_score, original_type5_score, original_type6a_score, 
                         original_type6b_score, original_type7_score, original_type8_score]

for i, scores in enumerate(original_types_scores):
    axes[0,0].plot(scores, label=original_types[i], color=original_colors[i], lw=1)

axes[0,0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[0,0].set_title("Original Individual Lines", fontsize=18)
axes[0,0].set_xlim(0, 25)
axes[0,0].legend(loc= "lower right")
axes[0,0].spines["top"].set_visible(False)
axes[0,0].spines["right"].set_visible(False)


#Graph 2: A single line graph of learning curve for "original_percept".
data_points = list(range(0,96)) 
original_total_types_scores = (original_type1_score + original_type2_score + original_type3_score +
                               original_type4_score + original_type5_score + original_type6a_score + 
                               original_type6b_score + original_type7_score + original_type8_score)

axes[0,1].plot(data_points, original_total_types_scores, "o-", label="Original Verbs", color="red", markersize=2, lw=1)
axes[0,1].set_title("Original Single-Line", fontsize=18)
axes[0,1].set_ylim(0.5, 1)
axes[0,1].set_xlim(0, 96)
axes[0,1].legend(loc= "lower right")
axes[0,1].spines["top"].set_visible(False)
axes[0,1].spines["right"].set_visible(False)

#Demarcating the types
axes[0,1].axvline(x=9, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=14, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=30, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=38, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=60, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=64, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=70, color="gray", linestyle="--", lw=1)
axes[0,1].axvline(x=88, color="gray", linestyle="--", lw=1)


#Graph 3: Individual learning curves for "high_start_percept".
high_start_colors = ['g', 'b', 'r', 'y', 'orange', 'k', 'purple', 'm', 'c'] 
high_start_types = ['Type 2', 'Type 6a', 'Type 6b', 'Type 1', 'Type 7', 'Type 3', 'Type 5', 'Type 4', 'Type 8']
high_start_type_scores = [high_start_type2_score, high_start_type6a_score, high_start_type6b_score, 
                          high_start_type1_score, high_start_type7_score, high_start_type3_score, 
                          high_start_type5_score, high_start_type4_score, high_start_type8_score]

for i, scores in enumerate(high_start_type_scores):
    axes[1,0].plot(scores, label=high_start_types[i],color=high_start_colors[i], lw=1)

axes[1,0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[1,0].set_title("High Start Individual Lines", fontsize=18)
axes[1,0].legend(loc= "lower right")
axes[1,0].set_xlim(0, 25)
axes[1,0].spines["top"].set_visible(False)
axes[1,0].spines["right"].set_visible(False)


#Graph 4: A single line graph of learning curve for "high_start_percept".
data_points = list(range(0,96)) 
total_high_start_scores = (high_start_type2_score + high_start_type6a_score + high_start_type6b_score + 
                           high_start_type1_score + high_start_type7_score + high_start_type3_score +
                           high_start_type5_score + high_start_type4_score + high_start_type8_score)

axes[1,1].plot(data_points, total_high_start_scores, "o-", label="High Start Verbs", color="red", markersize=2, lw=1)         
axes[1,1].set_title("High Start Single-Line", fontsize=18)
axes[1,1].set_ylim(0.5, 1)
axes[1,1].set_xlim(0, 96)
axes[1,1].legend(loc= "lower right")
axes[1,1].spines["top"].set_visible(False)
axes[1,1].spines["right"].set_visible(False)

#Demarcating the types
axes[1,1].axvline(x=5, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=9, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=15, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=24, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=42, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=58, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=80, color="gray", linestyle="--", lw=1)
axes[1,1].axvline(x=88, color="gray", linestyle="--", lw=1)


#Graph 5: Individual learning curves for "low_start_percept".
low_start_colors = ['c', 'm', 'purple', 'k', 'orange', 'y', 'r', 'b', 'g']
low_start_types = ['Type 8', 'Type 4', 'Type 5', 'Type 3', 'Type 7', 'Type 1', 'Type 6b', 'Type 6a', 'Type 2']
low_start_type_scores = [low_start_type8_score, low_start_type4_score, low_start_type5_score, 
                          low_start_type3_score, low_start_type7_score, low_start_type1_score, 
                          low_start_type6b_score, low_start_type6a_score, low_start_type2_score]

for i, scores in enumerate(low_start_type_scores):
    axes[2,0].plot(scores, label=low_start_types[i],color=low_start_colors[i], lw=1)

axes[2,0].set_xlabel("Trials", fontsize=14)
axes[2,0].set_ylabel("Correct Wickelfeatures Ratio", fontsize=14)
axes[2,0].set_title("Low Start Individual Lines", fontsize=18)
axes[2,0].legend(loc= "lower right")
axes[2,0].set_xlim(0, 25)
axes[2,0].spines["top"].set_visible(False)
axes[2,0].spines["right"].set_visible(False)


#Graph 6: A single-line graph showing the learning curves for "low_start_percept".
data_points = list(range(0,96)) 
total_low_start_scores = (low_start_type8_score + low_start_type4_score + low_start_type5_score + 
                         low_start_type3_score + low_start_type7_score + low_start_type1_score +
                         low_start_type6b_score + low_start_type6a_score + low_start_type2_score)

axes[2,1].plot(data_points, total_low_start_scores, "o-", label="Low Start Verbs", color="red", markersize=2, lw=1)         
axes[2,1].set_xlabel("Trials", fontsize=14)
axes[2,1].set_title("Low Start Single-Line", fontsize=18)
axes[2,1].set_ylim(0.5, 1)
axes[2,1].set_xlim(0, 96)
axes[2,1].legend(loc= "lower right")
axes[2,1].spines["top"].set_visible(False)
axes[2,1].spines["right"].set_visible(False)

#Demarcating the types
axes[2,1].axvline(x=8, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=16, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=38, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=43, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=72, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=81, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=87, color="gray", linestyle="--", lw=1)
axes[2,1].axvline(x=91, color="gray", linestyle="--", lw=1)


plt.suptitle("Learning Curves for Original Verbs VS Starting Score Rearranged Verbs", fontsize=18)

plt.show()

'''
Discuss your findings, which means STATE the model output, STATE why the model does it, STATE whether or not you believe it, and STATE why you do/don't believe it (500 words max).
Results
Criterion 1:
 - The high to low rearranged model clearly exhibits a U-shaped pattern as each new type is trained. 
 - There is a tendency for the starting and final scores to cluster together. Even without an upward curve in learning, scores tend to loop within a certain range.
 - A significant drop in scores is observed at the high to low rearranged model type 8. 
 - The low to high rearranged model also clearly shows a U-shaped pattern with each new type trained. 
 - Compared to the high to low rearranged model, there is a tendency for scores to increase as more traininging occurs. 

Criterion 2: 
 - Similar to the high to low rearranged model in Criterion 1, scores are closely clustered.
 - The high to low rearranged model, there are no notably drastic score changes such as type 8 in criterion 1 The high to low rearranged model, also no clear evidence of score improvement. 
 - The low to high rearranged model has the most evenly distributed scores among the four models, with a slight upward curve trend. 
Discussion
Following the analysis of four models, it appears that the outcomes for the first model and those based on pattern modelling did not meet expectations. This discrepancy was primarily due to the introduction of new types, which led to a curve progression characterized by sharp declines rather than smooth transitions. This trend was particularly pronounced in models trained from high to low values, regardless of the criteria used.

Neither criterion yielded significant results in the models trained from high to low values. Contrary to our hypothesis, these models appeared too independent, and the outcomes were less favourable than those observed in the original graph.

Conversely, models trained from low to high values demonstrated a sense of learning, especially the model that progressed from a low starting point to a high starting point, closely resembling the shape of the first model.

As a result, I consider the second model to be less reliable. The primary reason for these outcomes is believed to be the lack of flexibility in the total sample set (9 verb types).
The first model, which categorizes words solely based on frequency, allows for simultaneous exposure to a variety of verb forms. However, the second model, which clustered verbs with similar transformations, significantly lacked the flexibility or randomness necessary to handle the transformations of unlearned words. This led to a drop in scores with the introduction of new word types across all four models of the second approach. Therefore, the use of type-specific word lists was deemed an unsuitable method for data handling in this model.

For more ideal modelling, it would be necessary to standardise the number of words per type in the second dataset (9 verb types), ensure control for similarities in verb transformations across types, and standardise the data accordingly. Furthermore, quickly acquiring the most common words(high-frequency) early on to stabilise the model efficiently, followed by enhancing understanding and prediction accuracy of verb transformations through type-specific learning, would be ideal.
'''


#Exercise 7
#Experimental Predictions
#Construct and convert the novel verbs
'''
   Verb   Regular Past Irregular Past

1. plint, plinted , plent 
2. drouge, drouged , drowge 
3. frint, frinted , frent
4. clive, clived, clove
5. flarve, flarved, flurve
6. snibe, snibed, snobe
7. blave, blaved, blove
8. trink, trinked, trank
9. miple, mipled, mople 
10. quasp, quasped, quaspt
11. drindle, drindled, drondle 
12. flidge, flidged, fledge 
13. snulf, snulfed, snolf 
14. frask, frasked, frisk 
15. throve, throved, thrave
16. blenze, blenzed, blunze
17. crenke, crenked, crunke
18. dwimbe, dwimbed, dwombe
19. glishe, glished, glushe
20. pranke, pranked, prunke
'''

novel_verbs = ['plint', 'drouge', 'frint', 'clive', 'flarve', 'snibe', 'blave', 'trink', 'miple', 'quasp', 'drindle', 'flidge', 'snulf', 'frask', 'throve', 'blenze', 'crenke', 'dwimbe', 'glishe', 'pranke']
novel_regular_past_verb = ['plinted', 'drouged', 'frinted', 'clived', 'clived', 'flarved', 'snibed', 'blaved', 'trinked', 'mipled', 'quasped', 'drindled', 'flidged', 'snulfed', 'frasked', 'throved', 'blenzed', 'crenked', 'dwimbed', 'glished', 'pranked']
novel_irregular_past_verb = ['plent', 'drowge', 'clove', 'flurve', 'snobe', 'blove', 'trank', 'mople', 'quaspt', 'drondle', 'fledge', 'snolf', 'frisk', 'thrave', 'blunze', 'crunke', 'dwombe', 'glushe', 'prunke']
novel_bases = ['plint', 'drWj', 'frint', 'klIv', 'flarv', 'snIb', 'blAv', 'trink', 'mIpl', 'kwasped', 'drindl', 'flidj', 'sn*lf', 'frask', 'TrOv', 'blenz', 'krenk', 'dwImb', 'gliS', 'prank']
novel_regular_pasts =['plinted', 'drWjd', 'frinted', 'klIvd', 'flarvd', 'snIbd', 'blAvd', 'trinked', 'mIpld', 'kwaspd', 'drindld', 'flidjd', 'sn*ulfed', 'frasked', 'TrOvd', 'blenzd', 'krenkd', 'dwImbd', 'gliSd', 'prankd']
novel_irregular_pasts = ['plent', 'drOj', 'frent', 'klOv', 'flerv', 'snob', 'blOv', 'trank', 'mopl', 'kwaspt', 'dr*ndl', 'fledj', 'snolf', 'frisk', 'TrAv', 'bl*nz', 'kr*nk', 'dwomb', 'gl*S', 'pr*nk']


#Converting the phonemes into wickelfeatures. 
base_wickel_novel = np.array([activate_word(w) for w in novel_bases]).T
past_wickel_novel_regular = np.array([activate_word(w) for w in novel_regular_pasts]).T
past_wickel_novel_irregular = np.array([activate_word(w) for w in novel_irregular_pasts]).T


#Score each of the novel verb regular and irregular pasts before training
#Use a trained RM perceptron model(The frequency model): percept = Perceptron(active=rm_activation_function

scores_novel_regular_pasts_before = []
scores_novel_irregular_pasts_before = []
for i in range(len(novel_verbs)):
    scores_novel_regular_pasts_before.append(np.mean(percept.score(base_wickel_novel, past_wickel_novel_regular)))
    scores_novel_irregular_pasts_before.append(np.mean(percept.score(base_wickel_novel, past_wickel_novel_irregular)))
    print(scores_novel_regular_pasts_before[-1], scores_novel_irregular_pasts_before[-1])
    
    
#Mean of the score of the novel verb regular and irregular pasts before training

novel_regular_score_before = np.mean(percept.score(base_wickel_novel, past_wickel_novel_regular))
novel_irregular_score_before = np.mean(percept.score(base_wickel_novel, past_wickel_novel_irregular))

print(novel_regular_score_before)
print(novel_irregular_score_before)


#Learn and score the novel verb regular and irregular pasts
#Use a trained RM perceptron model(The frequency model): percept = Perceptron(active=rm_activation_function)

scores_novel_regular_pasts_after = []
scores_novel_irregular_pasts_after = []   

for i in range(len(novel_verbs)):
    percept.learn(base_wickel_novel[:,i, np.newaxis], past_wickel_novel_regular[:,i,np.newaxis])
    percept.learn(base_wickel_novel[:,i, np.newaxis], past_wickel_novel_irregular[:,i,np.newaxis])
    scores_novel_regular_pasts_after.append(np.mean(percept.score(base_wickel_novel, past_wickel_novel_regular)))
    scores_novel_irregular_pasts_after.append(np.mean(percept.score(base_wickel_novel, past_wickel_novel_irregular)))
    print(scores_novel_regular_pasts_after[-1], scores_novel_irregular_pasts_after[-1])
    

#Mean of the score of the novel verb regular and irregular pasts after training
novel_regular_score_after = np.mean(percept.score(base_wickel_novel, past_wickel_novel_regular))
novel_irregular_score_after = np.mean(percept.score(base_wickel_novel, past_wickel_novel_irregular))

print(novel_regular_score_after)
print(novel_irregular_score_after)


#plot the novel verbs data
fig, axes = plt.subplots(1, 2, figsize=(18,10), sharey=True)

#Graph 1:The bar graph of comparison the mean score before and after training
labels = ['Regular', 'Irregular']
before_training_scores = [novel_regular_score_before, novel_irregular_score_before]
after_training_scores =  [novel_regular_score_after, novel_irregular_score_after]
label_position = np.arange(len(labels))
bar_width = 0.4

axes[0].bar(label_position - bar_width/2, before_training_scores, label='Before Training', color='g', width=0.4)
axes[0].bar(label_position + bar_width/2, after_training_scores, label='After Training', color='orange', width= 0.4)

axes[0].set_xlabel("Novel Verbs", fontsize=14)
axes[0].set_ylabel("Score", fontsize=14)
axes[0].set_title("Mean Score Comparison Before and After Training", fontsize=16)
axes[0].set_xticks(label_position)
axes[0].set_xticklabels(labels)
axes[0].set_ylim([0.88,0.98])
axes[0].set_yticks(np.arange(0.88, 0.98, 0.01))
axes[0].legend(loc="upper left")


#Graph 2: The line graph of each score of the novel verbs before and after training
data_points = list(range(1,21))
axes[1].plot(data_points, scores_novel_regular_pasts_before, "o-", label="Before Training Regular Verbs", color="g", markersize=5, lw=1)
axes[1].plot(data_points, scores_novel_irregular_pasts_before, "o-", label="Before Training Irregular Verbs", color="g", markersize=5, lw=1, linestyle ='--')
axes[1].plot(data_points, scores_novel_regular_pasts_after, "o-", label='After Training Regular Verbs', color="orange", markersize=5, lw=1)
axes[1].plot(data_points, scores_novel_irregular_pasts_after, "o-", label='After Training Irregular Verbs', color="orange", markersize=5, lw=1, linestyle ='--')

axes[1].set_xlabel("Novel Verbs", fontsize=14)
axes[1].set_title("Score Comparison of Each Novel Verbs Before and After Training", fontsize=16)
axes[1].legend(loc="upper left")
axes[1].set_xticks(range(1, 21, 1))
axes[1].set_ylim([0.88,0.98])

plt.show()


'''
-Discuss the predictions, which means for each verb STATE which past tense form is most predicted by the model, STATE if you find the model's prediction surprising and STATE why you think the model made that prediction

Before the discussion, there were aspects that were overlooked. Firstly, it was necessary to identify the types of irregular past tense forms for novel verbs, which was not accomplished. Secondly, despite being aware of the 'predict' function within the perceptron class, it was omitted due to the scores being unexpectedly strange and the inability to understand how it worked.

The before training score can be considered a prediction since it was evaluated using the frequency-based model. Interestingly, similar to the original model, the score for irregular verbs was slightly higher.

The after training score reflects the evaluation after training the first model on these novel verbs. Notably, there was a significant improvement in the accuracy of irregular verbs after training.

Surprisingly, as mentioned, only the scores for after trained irregular verbs increased sharply, while the scores for regular verbs did not show significant changes. However, it seems plausible that the issue might be related to overfitting or underfitting, given that only one set of values exhibits significant increment. Instead, it might be because irregular past tense forms, lacking specific patterns or regularities, cause the model to allocate more weight or process them more carefully, leading to a sensitive reaction. 

Conversely, regular verbs, with their simple and easily discernable rules, may not engage the model's learning mechanisms as intensely, hence the stagnant scores. Therefore, it's plausible to think that the model reacts sensitively to the highly stimulating data of non-existent irregular past tense forms, resulting in these outcomes.



-Based on the model predictions, discuss if this is a good experiment for testing whether rules or similarity underly the generation of past tense forms

To summarize, this experiment is significantly suitable for testing irregular past tense forms. As described earlier, it was observed that the accuracy for irregular past tense forms sharply increased. According to the original paper, while regular verbs steadily rise in accuracy, irregular verbs initially draw a small U-shaped curve in their accuracy increase, but do not reach the absolute accuracy levels of regular verbs. However, the model in this assignment has shown that the accuracy for all irregular past tense forms surpassed that of regular verbs. Although the exact mechanism behind this outcome is not fully understood, it's clear that a suitable model for learning irregular past tense forms has been developed, as proven by the performance in the hypothetical irregular past tense test in Exercise 7.
'''
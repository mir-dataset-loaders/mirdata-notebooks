```
Encoding Scheme for Monophonic Sounds

A - Instrument Type
    B - Bass
    G - Guitar

B - Instrument Setting
    1 - Yamaha BB604, 1st setting
    2 - Yamaha BB604, 2nd setting
    3 - Warwick Corvette $$, 1st setting
    4 - Warwick Corvette $$, 2nd setting
    5 - not assigned
    6 - Schecter Diamond C-1 Classic, 1st setting
    7 - Schecter Diamond C-1 Classic, 2nd setting
    8 - Chester Stratocaster, 1st setting
    9 - Chester Stratocaster, 1st setting

C - Playing Technique
    1 - Finger plucking normal/soft
    2 - Finger plucking loud
    3 - Pick plucking

D - MIDI Number of Pitch
    Two digits between 28 and 76

E - String Number
    1 - E
    2 - A
    3 - D
    4 - G
    5 - B
    6 - E

F - Fret Number
    Two digits between 00 and 12

G - Effect Group
    1 - No Effect
    2 - Spatial Effect
    3 - Modulation Effect
    4 - Distortion Effect

H - Individual Effect
    11 - No Effect
    12 - No Effect, amplifier simulation
    21 - Feedback Delay
    22 - Slapback Delay
    23 - Reverb
    31 - Chorus
    32 - Flanger
    33 - Phaser
    34 - Tremolo
    35 - Vibrato
    41 - Distortion
    42 - Overdrive

I - Effect Setting
    1, 2, or 3

K - Identification Number
    Five digits, sequential with leading zeros

Example
B11-28100-3311-00625
ABC-DDEFF-GHHI-KKKKK
--> A-C: Bass, Yamaha, 1st setting, finger plucking normal/soft
    D-F: MIDI no. 28, E string, 0th fret (open string) --> low E
    G-I: Modulation effect, Chorus, 1st setting
    K  : Identification number

Encoding Scheme for Polyphonic Sounds

A - Instrument Type
    G - Guitar

B - Instrument Setting
    6 - Schecter Diamond C-1 Classic, 1st setting
    9 - Chester Stratocaster, 1st setting

C - Playing Technique
    4 - Pick plucking, intervals
    5 - Pick plucking, triads and four-note chords

D - MIDI Number of Pitch
    Two digits between 43 and 57

E - Polyphony Type
    11 - Minor third
    12 - Major third
    13 - Perfect fourth
    14 - Perfect fifth
    15 - Minor seventh
    16 - Major seventh
    17 - Octave
    21 - Major triad
    22 - Minor triad
    23 - Sus4 triad
    24 - Power chord
    25 - Major seventh chord
    26 - Dominant seventh chord
    27 - Minor seventh chord

F - Not assigned, always 0

G - Effect Group
    1 - No Effect
    2 - Spatial Effect
    3 - Modulation Effect
    4 - Distortion Effect

H - Individual Effect
    11 - No Effect
    12 - No Effect, amplifier simulation
    21 - Feedback Delay
    22 - Slapback Delay
    23 - Reverb
    31 - Chorus
    32 - Flanger
    33 - Phaser
    34 - Tremolo
    35 - Vibrato
    41 - Distortion
    42 - Overdrive

I - Effect Setting
    1, 2, or 3

K - Identification Number
    Five digits, sequential with leading zeros

Example
P64-43110-3311-46225
ABC-DDEEF-GHHI-KKKKK
--> A-C: Guitar, Schecter, pick plucking interval
    D-F: MIDI no. 43, minor third
    G-I: Modulation effect, Chorus, 1st setting
    K  : Identification number

Summarizing the Information in XML Lists

Each list includes information about all audio files from one instrument with a 
specific setting that has been processed with an audio effect in a specific setting. 
Each list contains a node with information about the audio effect and N additional 
nodes with information about the audio files. 
The information is stored in child nodes, whose node names are self-explanatory. 
As with the audio files, the relevant information is encoded in the filenames of the XML file.

Encoding Scheme for Lists

A-C - Instrument information, as with the audio files
D-F - Not assigned, always 0
G-I - Effect information, as with the audio files
K   - Identification number, three digits, sequential with leading zeros

Example
B11-00000-3311-013
ABC-DDEEF-GHHI-KKK
--> A-C: Guitar, Schecter, finger plucking
    D-F: Not significant
    G-I: Modulation effect, Chorus, 1st setting
    K  : Identification number
```
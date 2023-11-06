# ChordGEN
use seq2seq to generate chord from melody

USE STEP:

1.open seq2seq

2.change filepath in prepare_data.py and then run it

3.run transformer.py

NOTE: 

a midi file with only melody called "test.midi" should be created before, which is used to create the correspond chord.


INTRODUCTION:

In this music generation project, our primary objective is to develop a deep learning model capable of generating harmonically rich chord progressions that complement given melodies. The core goals of this project are as follows:

Melody-Chord Harmony Generation: Create a model that can take a melody as input and generate harmonically suitable chord progressions to accompany the melody. This harmony should enhance the musical quality of the given melody.

Transformer-Based Approach: Implement a sequence-to-sequence model with a transformer architecture, which has been proven effective in various natural language processing tasks, to handle the complex relationship between melody and harmony in music.

Data Processing and Preparation: Develop robust data processing and preparation pipelines that enable us to convert MIDI files into sequences of musical tokens suitable for training the model.

Training and Evaluation: Train the model on a dataset of melody-harmony pairs and establish an evaluation framework to assess the quality of generated harmonies.

User Interface for Music Generation: Users can input melodies, and the model generates corresponding harmonies in a musical format.

MIDI File Generation: Develop the capability to generate MIDI files that combine the input melodies with the generated harmonies.

Improvisation Support: Investigate the possibility of enhancing the model's ability to generate musical improvisations based on given melodies.

Documentation and Sharing: Provide comprehensive documentation of the project, including code, usage instructions, and explanations, to allow for easy sharing and future development.

By achieving these goals, we aim to create a versatile and user-friendly tool for music composers and enthusiasts, enabling them to explore new musical ideas and enhance their compositions with harmonically rich chord progressions.

The project combines concepts from deep learning, music theory, and data processing, ultimately promoting the intersection of artificial intelligence and music composition. Through the use of transformer-based models, we intend to achieve significant advancements in AI-generated music that can assist and inspire musicians and composers.







![embedding32 batch32](https://github.com/BOURBON0/ChordGEN/assets/54803330/cec08048-00b4-4a3d-bd01-67768ed7aac4)

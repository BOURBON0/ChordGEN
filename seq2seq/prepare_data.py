import os
import music21 as ms21
from tqdm import tqdm
import random

filepath = '../MIDI'

random.seed(42)

def divide_dataset(encode_seq, decode_seq):
    '''
    filepath: the path of folder MIDI
    divide the train and the dev
    '''
    encode_seq = [' '.join(sublist) for sublist in encode_seq]
    decode_seq = [' '.join(sublist) for sublist in decode_seq]

    total_size = len(encode_seq)
    train_size = int(0.9 * total_size)
    test_size = int(0.1 * total_size)

    indices = list(range(total_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    train_inp = [encode_seq[i] for i in train_indices]
    train_targ = [decode_seq[i] for i in train_indices]

    test_indices = indices[train_size:]
    test_inp = [encode_seq[i] for i in test_indices]
    test_targ = [decode_seq[i] for i in test_indices]
    
    with open(filepath + '/midi_train.txt','w',encoding='utf-8') as fp:
        for i in range(len(train_inp)):
            thisline = train_inp[i] + '\t' + train_targ[i] +'\r'
            fp.write(thisline)

    with open(filepath + '/midi_test.txt','w',encoding='utf-8') as fp:
        for i in range(len(test_inp)):
            thisline = test_inp[i] + '\t' + test_targ[i] +'\r'
            fp.write(thisline)

    return 0

def get_midi_path(filepath):
    '''
    filepath: the path of folder MIDI
    return all root of midis
    '''
    midi_path_list = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            midi_path = os.path.join(root, file)
            if os.path.splitext(midi_path)[1] == '.mid':
                midi_path_list.append(midi_path)
    return midi_path_list

def extract_melody_vector(midi_file):
    '''
    midi_file: root of one midi melody
    return the root and the vector of the midi
    '''
    score = ms21.converter.parse(midi_file)

    melody_vector = []
    for part in score.parts:
        for measure in part.getElementsByClass(ms21.stream.Measure):
            time_signature = measure.getTimeSignatures()[0] if measure.getTimeSignatures() else None
            notes = measure.flat.notesAndRests
            measure_info = []
            if time_signature:
                measure_info.append(time_signature.ratioString)
            for note in notes:
                if note.isRest:
                    measure_info.extend(['r', note.duration.quarterLength])
                elif note.isChord:
                    highest_note = max(note.notes, key=lambda x: x.pitch.midi)
                    measure_info.extend([highest_note.pitch.nameWithOctave, highest_note.duration.quarterLength])
                else:
                    measure_info.extend([note.pitch.nameWithOctave, note.duration.quarterLength])
            melody_vector += measure_info
    return midi_file, melody_vector

def extract_chord_vector(midi_file):
    '''
    midi_file: root of one midi chord
    return the root and the vector of the midi
    '''
    score = ms21.converter.parse(midi_file) 
    vector_list = [] 
    for part in score.parts:
        for measure in part.getElementsByClass(ms21.stream.Measure):
            time_signature = measure.getTimeSignatures()[0] if measure.getTimeSignatures() else None
            notes_and_chords = measure.flat.notesAndRests
            measure_info = []

            if time_signature:
                measure_info.append(time_signature.ratioString)

            for note in notes_and_chords:
                if note.isRest:
                    measure_info.extend(['r', note.duration.quarterLength])
                elif note.isChord:
                    root_pitch = note.root().nameWithOctave
                    chord_name = note.commonName
                    chord_name = str(chord_name).replace(" ", "")
                    measure_info.extend([chord_name, root_pitch, note.duration.quarterLength])

            vector_list += measure_info

    return midi_file, vector_list

def create_dataset(filepath):
    '''
    filepath: the path of folder MIDI
    return:
        encode_seq: include all melody list
        decode_seq: include all chord list
    '''
    melodyans = {}
    chordans = {}
    melodylist = get_midi_path(filepath+'/melody')
    chordlist = get_midi_path(filepath+'/chords')
    progress_bar = tqdm(total=len(melodylist), desc="load melodypiece now", unit="item")
    for melodypiece in melodylist:
        melodyaddress, melody_vector = extract_melody_vector(melodypiece)
        melodyans[melodyaddress[41:]] = melody_vector
        progress_bar.update(1)
    progress_bar.close()

    progress_bar = tqdm(total=len(chordlist), desc="load chordpiece now", unit="item")
    for chordpiece in chordlist:
        chordaddress, chord_vector = extract_chord_vector(chordpiece)
        chordans[chordaddress[41:]] = chord_vector
        progress_bar.update(1)
    progress_bar.close()
    encode_seq = []
    decode_seq = []
    melody_seq = {k1:melodyans[k1] for k1 in melodyans if k1 in chordans}
    chord_seq = {k2:chordans[k2] for k2 in chordans if k2 in melodyans}
    count = 0
    counti = 0
    for melody_seq, chord_seq in zip(melody_seq.values(), chord_seq.values()):
        if len(melody_seq) > 1000:
            melody_seq = melody_seq[:1000]
            count += 1
        if len(chord_seq) > 350:
            counti += 1
            chord_seq = chord_seq[:350]
        melody_seq = [str(item) for item in melody_seq]
        chord_seq = [str(item) for item in chord_seq]
        encode_seq.append(melody_seq)
        decode_seq.append(chord_seq)
    print(count,counti)
    return encode_seq, decode_seq

if __name__ == '__main__':
    
    encode_seq, decode_seq = create_dataset(filepath)
    divide_dataset(encode_seq, decode_seq)
    
    
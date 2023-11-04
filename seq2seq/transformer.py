import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import os
import music21 as ms21
from torch import Tensor
from prepare_data import extract_melody_vector, filepath
from data_loader import en_total_words, cn_total_words, train_dataloader, test_dataloader, cn_bos_idx, \
    cn_eos_idx, decode_sents, answer_sents, PAD_IDX, UNK_IDX, BOS, EOS, cn_dict, en_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chordkind = {'augmentedtriad':[0,4,8],
             'Germanaugmentedsixthchord':[0,4,7,10],
             'enharmonicequivalenttodiminishedtriad':[0,3,6],
             'half-diminishedseventhchord':[0,3,6,10],
             'flat-ninthpentachord':[0,4,6,9,13],
             'majortriad':[0,4,7],
             'enharmonictodominantseventhchord':[0,4,7,10],
             'enharmonicequivalenttominortriad':[0,3,7],
             'diminishedtriad':[0,3,6],
             'enharmonicequivalenttomajortriad':[0,4,7],
             'minortriad':[0,3,7],
             'minorseventhchord':[0,3,7,10],
             'augmentedseventhchord':[0,4,8,10],
             'dominantseventhchord':[0,4,7,10]}
        
def generate_square_subsequent_mask(sz):
 
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    # ignore pad
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):

    def __init__(self, emb_size: int, max_len: int = 5000, dropout: float = 0.2):
        super(PositionalEncoding, self).__init__()
        # Position Encoding
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_emb = torch.zeros((max_len, emb_size))
        pos_emb[:, 0::2] = torch.sin(pos * den)
        pos_emb[:, 1::2] = torch.cos(pos * den)
        pos_emb = pos_emb.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, token_emb: Tensor):
        return self.dropout(token_emb + self.pos_emb[:token_emb.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(device)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2Seq(nn.Module):
  
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size,
                 tgt_vocab_size, dim_feedforward: int = 512, dropout: float = 0.2):
        super(Seq2Seq, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_token_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_token_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_token_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_token_emb(tgt))

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_token_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_token_emb(tgt)), memory, tgt_mask)

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


EMB_SIZE = 64

model = Seq2Seq(3, 3, emb_size=EMB_SIZE, nhead=8, src_vocab_size=en_total_words, tgt_vocab_size=cn_total_words,
                dim_feedforward=EMB_SIZE)

model.init()

model.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-9)


def train_epoch():
    
    model.train()
    losses = 0
    
    for i, (src, src_lens, tgt, tgt_lens) in enumerate(train_dataloader):

        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)
        # input:[BOS,w1,w2] -> output:[w1,w2,BOS]
        tgt_input = tgt[:-1, :]
        tgt_input = tgt_input.to(device)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        if i % 200 == 0:
            # print("loss:",loss.item())
            testing_loss = test_translate()

    return losses / len(train_dataloader), testing_loss


def translate(src, src_mask, max_len=1000):
    
    memory = model.encode(src, src_mask)
    batch_size = src.size(1)
    ys = torch.ones(1, batch_size).fill_(cn_bos_idx).type(torch.long)
    ys = ys.to(device)
    
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        tgt_mask = tgt_mask.to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
    return ys


@torch.no_grad()
def test_translate():
    
    model.eval()
    model.to(device)
    total_loss = 0
    num_samples = 0
    
    for i, (src, src_lens, tgt, tgt_lens) in enumerate(test_dataloader):
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)
        tgt_input = tgt[:-1, :]
        tgt_input = tgt_input.to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)
        # pred = translate(src, src_mask)
        # print(decode_sents(src.cpu().transpose(0, 1), False))
        # print(decode_sents(tgt.cpu().transpose(0, 1)))
        # print(decode_sents(pred.cpu().transpose(0, 1)))
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        total_loss += loss.item()
        num_samples += 1

        testing_loss = total_loss / num_samples
        
        return testing_loss

def translate_sentence(sentence, src_dict, tgt_dict, model, device, max_len=1000):
  
    # Tokenize the input sentence and add BOS and EOS tokens
    sentence = [BOS] + sentence.split() + [EOS]
    # Encode the input sentence using the source dictionary
    input_tensor = torch.tensor([src_dict.get(word, UNK_IDX) for word in sentence], dtype=torch.long).unsqueeze(1).to(device)
    input_length = input_tensor.size(0)

    # Create masks and initialize the translation output
    src_mask = generate_square_subsequent_mask(input_length).to(device)
    tgt = torch.ones(1, 1).fill_(cn_bos_idx).type(torch.long).to(device)

    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).type(torch.bool).to(device)
        out = model.decode(tgt, model.encode(input_tensor, src_mask), tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        tgt = torch.cat([tgt, next_word.unsqueeze(0)], dim=0)

        if next_word.item() == cn_dict[EOS]:
            break
    # Decode the translated sentence using the target dictionary
    tgt = tgt.cpu()
    translation = answer_sents(tgt.transpose(0, 1), is_cn=True)[0]
    return translation

def save_midi(predict_chord, existing_midi_file):
    
    chord_stream = ms21.stream.Stream()
    elements = predict_chord.split()
    current_measure = None
    istate = 0
    
    while istate < len(elements):
        
        if elements[istate] == 'r':
            rest = ms21.note.Rest()
            rest.duration.quarterLength = float(elements[istate+1])
            current_measure.append(rest)
            istate += 2

        elif '/' in elements[istate]:
            if istate != 0:
                chord_stream.append(current_measure)

            time_signature = ms21.meter.TimeSignature(elements[istate])
            current_measure = ms21.stream.Measure()
            current_measure.insert(0, time_signature)
            istate += 1

        elif elements[istate] in chordkind:
            pitchlist = chordkind[elements[istate]]
            root_obj = ms21.note.Note(elements[istate+1])
            rootmidi = root_obj.pitch.midi
            pitchlist = [i+rootmidi for i in pitchlist]
            now_chord = ms21.chord.Chord(pitchlist,quarterLength=float(elements[istate+2]))
            current_measure.append(now_chord)
            istate += 3
        else:
            istate += 1

    melody = ms21.converter.parse(existing_midi_file)
    melody.append(chord_stream)
    melody.write("midi", filepath + "/combined_output.mid")

    return chord_stream

if __name__ == '__main__':

    model_weights_file = filepath + "/transformer_epoch.model"

    if os.path.exists(model_weights_file):
        model_weights = torch.load(model_weights_file)
        model.load_state_dict(model_weights)
        model.eval()
        model.to(device)

# # train
    trainlosses = []
    testlosses = []
    for i in range(20):
        trainloss, testloss = train_epoch()
        trainlosses.append(trainloss)
        testlosses.append(testloss)
        print(f"[epoch {i+1}] training loss: {trainloss}", f"testing loss: {testloss}")
        torch.save(model.state_dict(), filepath+"/transformer_epoch.model")
    
    
    if os.path.exists(filepath + '/loss.txt'):
        trainlossbefore = []
        testlossbefore = []
        with open(filepath + '/loss.txt', "r", encoding='utf-8') as f:
            for line in f:
                parts = line.strip(' ').split('\t')
                trainlossbefore.append(float(parts[0]))
                testlossbefore.append(float(parts[1]))
        trainlosses = trainlossbefore + trainlosses
        testlosses = testlossbefore + testlosses

    with open(filepath + '/loss.txt', "w", encoding='utf-8') as f:
        for i in range(len(trainlosses)):
            if i != 0:
                f.write('\n')
            f.write(str(trainlosses[i])+'\t'+str(testlosses[i]))


    plt.plot(range(1, len(trainlosses) + 1), trainlosses, marker='o', color='blue', label='Train Loss')
    plt.plot(range(1, len(testlosses) + 1), testlosses, marker='o', color='red', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
    

    filen, melody = extract_melody_vector(filepath + '/test.mid')
    melody = [str(i) for i in melody]
    melody = ' '.join(melody)


    predict_chord = translate_sentence(melody, en_dict, cn_dict, model, device)
    print("Input: ", melody)
    print("Translation: ", predict_chord)
    save_midi(predict_chord, filen)
